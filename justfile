default:
    just --list

# pathlength 10000 npoints 4096
train fold pathlength npoints:
    python train_freseg.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}}

train_frenet fold pathlength npoints:
    python train_freseg.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}}

# pathlength 10000 npoints 4096
#train cuda fold pathlength npoints:
#    CUDA_VISIBLE_DEVICES={{cuda}} python train_freseg.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}}

#train_frenet cuda fold pathlength npoints:
#   CUDA_VISIBLE_DEVICES={{cuda}} python train_freseg.py --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}}
