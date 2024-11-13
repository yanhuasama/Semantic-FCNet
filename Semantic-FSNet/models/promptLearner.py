import os.path as osp
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
_tokenizer = _Tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            print(f"ctx_init_in: {ctx_init}")
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if args.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                print(f"Initial n_ctx: {n_ctx} (type: {type(n_ctx)})")
                print(f"Initial n_ctx: {ctx_dim} (type: {type(ctx_dim)})")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print(self.ctx.shape)
        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(f"Prompts: {prompts}")
        # print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_, _ = clip.load(args.backbone, device=device, jit=False)
        clip_model_.cuda()

        # prompts_ = [prompt_prefix + " " + name + "." for name in classnames]
        temp = "a photo of a {}."
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512)),
            ("relu", nn.ReLU(inplace=True))
            # ("linear2", nn.Linear(128, 512))
        ]))

        if args.prec == "fp16":
            self.meta_net.half()

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.class_token_position

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1).to(device)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone",
                        type=str,
                        default="ViT-B/16",
                        help="name of CNN backbone")
    args = parser.parse_args()

    clip_model, _ = clip.load(args.backbone, device=device, jit=False)

    classnames = ['house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'jellyfish', 'dugong', 'Walker_hound', 'Saluki', 'Gordon_setter', 'komondor', 'boxer', 'Tibetan_mastiff', 'French_bulldog', 'Newfoundland', 'miniature_poodle', 'Arctic_fox', 'ladybug', 'three-toed_sloth', 'rock_beauty', 'aircraft_carrier', 'ashcan', 'barrel', 'beer_bottle', 'carousel', 'chime', 'clog', 'cocktail_shaker', 'dishrag', 'dome', 'file', 'fire_screen', 'frying_pan', 'hair_slide', 'holster', 'lipstick', 'oboe', 'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'prayer_rug', 'reel', 'slot', 'snorkel', 'solar_dish', 'spider_web', 'stage', 'tank', 'tile_roof', 'tobacco_shop', 'unicycle', 'upright', 'wok', 'worm_fence', 'yawl', 'street_sign', 'consomme', 'hotdog', 'orange', 'cliff', 'bolete', 'ear', 'horizontal_bar', 'combination_lock', 'catamaran', 'poncho', 'miniskirt', 'Ibizan_hound', 'white_wolf', 'rhinoceros_beetle', 'garbage_truck', 'carton', 'iPod', 'meerkat', 'missile', 'cannon', 'goose', 'coral_reef', 'dalmatian', 'nematode', 'ant', 'black-footed_ferret', 'king_crab', 'lion', 'vase', 'golden_retriever', 'mixing_bowl', 'malamute', 'African_hunting_dog', 'cuirass', 'bookshop', 'crate', 'hourglass', 'electric_guitar', 'trifle', 'school_bus', 'theater_curtain', 'scoreboard']
    prompt_learner =PromptLearner(args,classnames,clip_model)
    prompts = prompt_learner()
    print(prompts.shape)



