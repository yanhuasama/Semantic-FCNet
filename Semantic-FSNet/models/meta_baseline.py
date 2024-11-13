import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import clip
import models
import utils
import math
from .models import register
from torchvision import transforms
from models.promptLearner import *

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
clip_model.eval()
seed = 66
torch.manual_seed(seed)
np.random.seed(seed)

mini_classnames = ['house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'jellyfish', 'dugong',
                   'Walker_hound', 'Saluki', 'Gordon_setter', 'komondor', 'boxer', 'Tibetan_mastiff', 'French_bulldog',
                   'Newfoundland', 'miniature_poodle', 'Arctic_fox', 'ladybug', 'three-toed_sloth', 'rock_beauty',
                   'aircraft_carrier', 'ashcan', 'barrel', 'beer_bottle', 'carousel', 'chime', 'clog',
                   'cocktail_shaker',
                   'dishrag', 'dome', 'file', 'fire_screen', 'frying_pan', 'hair_slide', 'holster', 'lipstick', 'oboe',
                   'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'prayer_rug', 'reel', 'slot', 'snorkel',
                   'solar_dish', 'spider_web', 'stage', 'tank', 'tile_roof', 'tobacco_shop', 'unicycle', 'upright',
                   'wok', 'worm_fence', 'yawl', 'street_sign', 'consomme', 'hotdog', 'orange', 'cliff', 'bolete', 'ear',
                   'horizontal_bar', 'combination_lock', 'catamaran', 'poncho', 'miniskirt', 'Ibizan_hound',
                   'white_wolf', 'rhinoceros_beetle', 'garbage_truck', 'carton', 'iPod', 'meerkat', 'missile', 'cannon',
                   'goose', 'coral_reef', 'dalmatian', 'nematode', 'ant', 'black-footed_ferret', 'king_crab', 'lion',
                   'vase', 'golden_retriever', 'mixing_bowl', 'malamute', 'African_hunting_dog', 'cuirass', 'bookshop',
                   'crate', 'hourglass', 'electric_guitar', 'trifle', 'school_bus', 'theater_curtain', 'scoreboard']

tiered_classnames = ['Yorkshire terrier', 'space shuttle', 'drake', 'plane', 'mosquito net', 'sax', 'container ship',
                     'patas', 'cheetah', 'submarine', 'prison', 'can opener', 'syringe', 'odometer', 'bassoon',
                     'Kerry blue terrier', 'scale', 'baseball', 'cassette player', 'shield', 'goldfinch', 'cornet',
                     'flute', 'stopwatch', 'basketball', 'brassiere', 'bulbul', 'steel drum', 'bolo tie',
                     'planetarium', 'stethoscope', 'proboscis monkey', 'guillotine', 'Scottish deerhound', 'ocarina',
                     'Border terrier', 'capuchin', 'magnetic compass', 'alligator lizard', 'baboon', 'sundial',
                     'gibbon', 'grand piano', 'Arabian camel', 'basset', 'corkscrew', 'miniskirt', 'missile',
                     'hatchet', 'acoustic guitar', 'impala', 'parking meter', 'greenhouse', 'home theater',
                     'hartebeest', 'hippopotamus', 'warplane', 'albatross', 'umbrella', 'shoe shop', 'suit',
                     'pickelhaube', 'soccer ball', 'yawl', 'screwdriver', 'Madagascar cat', 'garter snake', 'bustard',
                     'tabby', 'airliner', 'tobacco shop', 'Italian greyhound', 'projector', 'bittern', 'rifle',
                     'pay-phone', 'house finch', 'monastery', 'lens cap', 'maillot', 'canoe', 'letter opener', 'nail',
                     'guenon', 'CD player', 'safety pin', 'harp', 'disk brake', 'otterhound', 'green mamba', 'violin',
                     'American coot', 'ram', 'jay', 'trench coat', 'Indian cobra', 'projectile', 'schooner', 'magpie',
                     'Norwich terrier', 'cairn', 'crossword puzzle', 'snow leopard', 'gong', 'library',
                     'swimming trunks', 'Staffordshire bullterrier', 'Lakeland terrier', 'black stork', 'king penguin',
                     'water ouzel', 'macaque', 'lynx', 'ping-pong ball', 'standard schnauzer', 'Australian terrier',
                     'stupa', 'white stork', 'king snake', 'Airedale', 'banjo', 'Windsor tie', 'abaya', 'stole',
                     'vine snake', 'Bedlington terrier', 'langur', 'catamaran', 'sarong', 'spoonbill',
                     'boa constrictor', 'ruddy turnstone', 'hognose snake', 'American chameleon', 'rugby ball',
                     'black swan', 'frilled lizard', 'oscilloscope', 'ski mask', 'marmoset', 'Komodo dragon',
                     'accordion', 'horned viper', 'bookshop', 'Boston bull', 'crane', 'junco', 'silky terrier',
                     'Egyptian cat', 'Irish terrier', 'leopard', 'sea snake', 'hog', 'colobus', 'chickadee',
                     'Scotch terrier', 'digital watch', 'analog clock', 'zebra', 'American Staffordshire terrier',
                     'European gallinule', 'lampshade', 'holster', 'jaguar', 'cleaver', 'brambling', 'orangutan',
                     'combination lock', 'tile roof', 'borzoi', 'water snake', 'knot', 'window shade', 'mosque',
                     'Walker hound', 'cardigan', 'warthog', 'whiptail', 'plow', 'bluetick', 'poncho', 'shovel',
                     'sidewinder', 'croquet ball', 'sorrel', 'airship', 'goose', 'church', 'titi', 'butcher shop',
                     'diamondback', 'common iguana', 'Saluki', 'monitor', 'sunglasses', 'flamingo', 'seat belt',
                     'Persian cat', 'gorilla', 'banded gecko', 'thatch', 'beagle', 'limpkin', 'jigsaw puzzle', 'rule',
                     'hammer', 'cello', 'lab coat', 'indri', 'vault', 'cellular telephone', 'whippet', 'siamang',
                     'loupe', 'modem', 'lifeboat', 'dial telephone', 'cougar', 'thimble', 'ibex', 'lawn mower',
                     'bell cote', 'chain mail', 'hair slide', 'apiary', 'harmonica', 'green snake', 'howler monkey',
                     'digital clock', 'restaurant', 'miniature schnauzer', 'panpipe', 'pirate', 'window screen',
                     'binoculars', 'Afghan hound', 'cinema', 'liner', 'ringneck snake', 'redshank', 'Siamese cat',
                     'thunder snake', 'boathouse', 'jersey', 'soft-coated wheaten terrier', 'scabbard', 'muzzle',
                     'Ibizan hound', 'tennis ball', 'padlock', 'kimono', 'redbone', 'wild boar', 'dowitcher', 'oboe',
                     'electric guitar', 'trimaran', 'barometer', 'llama', 'robin', 'maraca', 'feather boa',
                     'Dandie Dinmont', 'Lhasa', 'bow', 'punching bag', 'volleyball', 'Norfolk terrier', 'Gila monster',
                     'fire screen', 'hourglass', 'chimpanzee', 'birdhouse', 'Sealyham terrier', 'Tibetan terrier',
                     'palace', 'wreck', 'overskirt', 'pelican', 'French horn', 'tiger cat', 'barbershop', 'revolver',
                     'Irish wolfhound', 'lion', 'fur coat', 'ox', 'cuirass', 'grocery store', 'hoopskirt',
                     'spider monkey', 'tiger', 'bloodhound', 'red-backed sandpiper', 'drum', 'radio telescope',
                     'West Highland white terrier', 'bow tie', 'golf ball', 'barn', 'binder', 'English foxhound',
                     'bison', 'screw', 'assault rifle', 'diaper', 'bighorn', 'Weimaraner', 'computer keyboard',
                     'black-and-tan coonhound', 'little blue heron', 'breastplate', 'gasmask', 'aircraft carrier',
                     'iPod', 'organ', 'wall clock', 'rock python', 'squirrel monkey', 'bikini', 'water buffalo',
                     'upright', 'chime', 'confectionery', 'indigo bunting', 'green lizard', 'Norwegian elkhound',
                     'dome', 'buckle', 'giant schnauzer', 'jean', 'wire-haired fox terrier', 'African chameleon',
                     'trombone', 'oystercatcher', 'sweatshirt', 'American egret', 'marimba', 'gazelle',
                     'red-breasted merganser', 'tape player', 'speedboat', 'gondola', 'night snake', 'cannon',
                     'plunger', 'balloon', 'toyshop', 'agama', 'fireboat', 'bakery', 'cab', 'jeep', 'English setter',
                     'flat-coated retriever', 'bassinet', 'sports car', 'golfcart', 'clumber', 'puck', 'reel',
                     'Welsh springer spaniel', 'car wheel', 'wardrobe', 'go-kart', 'switch', 'crib', 'laptop',
                     'thresher', 'web site', 'English springer', 'iron', 'Gordon setter', 'Labrador retriever',
                     'Irish water spaniel', 'amphibian', 'file', 'harvester', 'convertible', 'paddlewheel',
                     'microwave', 'swing', 'chiffonier', 'desktop computer', 'gas pump', 'beach wagon', 'carousel',
                     "potter's wheel", 'folding chair', 'fire engine', 'slide rule', 'vizsla', 'waffle iron',
                     'trailer truck', 'toilet seat', 'medicine chest', 'Brittany spaniel', 'Chesapeake Bay retriever',
                     'cash machine', 'moped', 'Model T', 'bookcase', 'ambulance', 'German short-haired pointer',
                     'dining table', 'minivan', 'police van', 'entertainment center', 'throne', 'desk', 'notebook',
                     'snowplow', 'cradle', 'abacus', 'hand-held computer', 'Dutch oven', 'toaster', 'barber chair',
                     'vending machine', 'four-poster', 'rotisserie', 'hook', 'vacuum', 'pickup', 'table lamp',
                     'rocking chair', 'prayer rug', 'moving van', 'studio couch', 'racer', 'park bench',
                     'Irish setter', 'refrigerator', 'china cabinet', 'cocker spaniel', 'radiator', 'Sussex spaniel',
                     'hand blower', 'slot', 'golden retriever', 'curly-coated retriever', 'limousine', 'washer',
                     'garbage truck', 'dishwasher', 'pinwheel', 'espresso maker', 'tow truck', 'Siberian husky',
                     'dung beetle', 'jackfruit', 'miniature pinscher', 'tiger shark', 'weevil', 'goldfish',
                     'schipperke', 'Tibetan mastiff', 'orange', 'whiskey jug', 'hammerhead', 'bull mastiff', 'eggnog',
                     'bee', 'tench', 'chocolate sauce', 'dragonfly', 'zucchini', 'kelpie', 'stone wall',
                     'butternut squash', 'mushroom', 'Old English sheepdog', 'dam', 'picket fence', 'espresso',
                     'beer bottle', 'plate', 'dough', 'sandbar', 'boxer', 'bathtub', 'beaker', 'bucket',
                     'Border collie', 'sturgeon', 'worm fence', 'seashore', 'long-horned beetle', 'turnstile',
                     'groenendael', 'vase', 'teapot', 'water tower', 'strawberry', 'burrito', 'cauliflower', 'volcano',
                     'valley', 'head cabbage', 'tub', 'lacewing', 'coral reef', 'hot pot', 'custard apple', 'monarch',
                     'cricket', 'pill bottle', 'walking stick', 'promontory', 'malinois', 'pizza', 'malamute',
                     'kuvasz', 'trifle', 'fig', 'komondor', 'ant', 'electric ray', 'Granny Smith', 'cockroach',
                     'stingray', 'red wine', 'Saint Bernard', 'ice lolly', 'bell pepper', 'cup', 'pomegranate',
                     'Appenzeller', 'hay', 'EntleBucher', 'sulphur butterfly', 'mantis', 'Bernese mountain dog',
                     'banana', 'water jug', 'cicada', 'barracouta', 'washbasin', 'wine bottle', 'Rottweiler', 'briard',
                     'puffer', 'ground beetle', 'Bouvier des Flandres', 'chainlink fence', 'damselfly', 'grasshopper',
                     'carbonara', 'German shepherd', 'guacamole', 'leaf beetle', 'caldron', 'fly', 'bannister',
                     'spaghetti squash', 'coffee mug', 'gar', 'barrel', 'eel', 'rain barrel', 'coho', 'water bottle',
                     'menu', 'tiger beetle', 'Great Dane', 'rock beauty', 'anemone fish', 'mortar', 'Eskimo dog',
                     'affenpinscher', 'breakwater', 'artichoke', 'broccoli', 'French bulldog', 'coffeepot', 'cliff',
                     'ladle', 'sliding door', 'leafhopper', 'collie', 'Doberman', 'pitcher', 'admiral',
                     'cabbage butterfly', 'geyser', 'cheeseburger', 'grille', 'ladybug', 'great white shark',
                     'pineapple', 'cardoon', 'pop bottle', 'lionfish', 'cucumber', 'face powder', 'Shetland sheepdog',
                     'ringlet', 'Greater Swiss Mountain dog', 'alp', 'consomme', 'potpie', 'acorn squash', 'ice cream',
                     'lakeside', 'hotdog', 'rhinoceros beetle', 'lycaenid', 'lemon']

cifar_classnames = ['apple', 'aquarium_fish', 'bear', 'bee', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'can', 'castle',
                    'caterpillar', 'chair', 'clock', 'cloud', 'cockroach', 'couch', 'cup', 'dinosaur', 'dolphin',
                    'elephant', 'forest', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lawn_mower', 'lion',
                    'lizard', 'lobster', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'palm_tree',
                    'pear', 'pine_tree', 'plate', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'seal',
                    'shrew', 'skunk', 'skyscraper', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'tank',
                    'tiger', 'train', 'trout', 'tulip', 'turtle', 'willow_tree', 'wolf', 'beaver', 'beetle',
                    'butterfly', 'camel', 'cattle', 'crab', 'crocodile', 'flatfish', 'lamp', 'maple_tree', 'motorcycle',
                    'otter', 'sea', 'shark', 'television', 'tractor', 'baby', 'bed', 'bicycle', 'chimpanzee', 'fox',
                    'leopard', 'man', 'pickup_truck', 'plain', 'poppy', 'rocket', 'rose', 'snail', 'sweet_pepper',
                    'table', 'telephone', 'wardrobe', 'whale', 'woman', 'worm']

fc100_classnames = ['apple', 'aquarium_fish', 'bed', 'bicycle', 'bottle', 'bowl', 'bridge', 'bus', 'can', 'castle',
                    'chair', 'clock', 'cloud', 'couch', 'crocodile', 'cup', 'dinosaur', 'flatfish', 'forest', 'house',
                    'keyboard', 'lamp', 'lawn_mower', 'lizard', 'maple_tree', 'motorcycle', 'mountain', 'mushroom',
                    'oak_tree', 'orange', 'orchid', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                    'poppy', 'ray', 'road', 'rocket', 'rose', 'sea', 'shark', 'skyscraper', 'snake', 'streetcar',
                    'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tractor', 'train',
                    'trout', 'tulip', 'turtle', 'wardrobe', 'willow_tree', 'bear', 'camel', 'cattle', 'chimpanzee',
                    'crab', 'elephant', 'hamster', 'kangaroo', 'leopard', 'lion', 'lobster', 'mouse', 'rabbit', 'shrew',
                    'snail', 'spider', 'squirrel', 'tiger', 'wolf', 'worm', 'baby', 'beaver', 'bee', 'beetle', 'boy',
                    'butterfly', 'caterpillar', 'cockroach', 'dolphin', 'fox', 'girl', 'man', 'otter', 'porcupine',
                    'possum', 'raccoon', 'seal', 'skunk', 'whale', 'woman']


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=8, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, args, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True, beta=0.5):
        super().__init__()
        if args is None:
            args = argparse.Namespace(backbone='default_backbone')
        self.encoder = models.make(encoder, **encoder_args)
        self.args = args
        self.method = method
        self.CrossAttention = Attention(embed_dim=512, hidden_dim=None, out_dim=None, n_head=8,
                                        score_function='bi_linear', dropout=0)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.1))
        if args.dataset == "mini":
            self.prompt_learner = PromptLearner(args, mini_classnames, clip_model)
        elif args.dataset == "tiered":
            self.prompt_learner = PromptLearner(args, tiered_classnames, clip_model)
        elif args.dataset == "cifar":
            self.prompt_learner = PromptLearner(args, cifar_classnames, clip_model)
        else:
            self.prompt_learner = PromptLearner(args, fc100_classnames, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.text_encoder = TextEncoder(clip_model)

        self.text_def_fea = torch.tensor(np.load(args.text_def_fea)).cuda()

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

        self.alpha = args.alpha
        self.beta = args.beta

    def freeze_prompt_learner(self):
        for name, param in self.prompt_learner.named_parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

    '''
    x_shot:[4,5,1,3,80,80]
    x_query:[4,75,3,80,80]

    '''

    def forward(self, epoch, labels, x_shot, x_query):
        prompts = self.prompt_learner()
        shot_shape = x_shot.shape[:-3]  # [4,5,1]
        query_shape = x_query.shape[:-3]  # [4,75]
        img_shape = x_shot.shape[-3:]  # [3,80,80]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot_fea, x_query_fea = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_query_fea = x_query_fea.view(*query_shape, -1)

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features[labels]
        if epoch > self.args.epoch:
            text_features = text_features * self.alpha + self.text_def_fea[labels] * (1 - self.alpha)

        fusion_fea, _ = self.CrossAttention(text_features.to(torch.float32), x_shot_fea.to(torch.float32))
        fusion_fea = fusion_fea.squeeze(dim=1)
        x_shot_refuse = self.fc(torch.cat([x_shot_fea, fusion_fea], dim=1))
        x_shot_refuse = x_shot_refuse.view(*shot_shape, -1)
        x_shot_fea = x_shot_fea.view(*shot_shape, -1)
        x_shot_refuse = x_shot_refuse * self.beta + x_shot_fea * (1 - self.beta)

        if self.method == 'cos':
            x_shot_refuse = x_shot_refuse.mean(dim=-2)
            x_shot_refuse = F.normalize(x_shot_refuse, dim=-1)
            x_query_fea = F.normalize(x_query_fea, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot_refuse = x_shot_refuse.mean(dim=-2)
            metric = 'sqr'

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_old = self.ori_embedding[labels]
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        logits = utils.compute_logits(
            x_query_fea, x_shot_refuse, metric=metric, temp=self.temp)
        return logits, score
