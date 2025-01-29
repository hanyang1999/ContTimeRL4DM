CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/DDPO_varying_steps_infr.py --config config/dgx.py:imagereward &
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch scripts/train_with_valuefunc_cont.py --config config/dgx_with_value_func.py:imagereward &
wait