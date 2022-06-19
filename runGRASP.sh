/home/ch222974/Desktop/conda_envs/dce_env/bin/python3.7 /home/ch222199/Documents/GIT/mocoRec/dce_mri/convert_raw_to_dataset.py -i /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/MRU021622.csv -o /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/ -fr True -fc True -nspkc 500 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ch222974/Desktop/conda_envs/dce_env/lib

/home/ch222974/Desktop/conda_envs/dce_env/bin/python3.7 /home/ch222199/Documents/GIT/mocoRec/dce_mri/demo_reconstruction.py -p /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/grasp_params_l0125.json -d /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/loader.csv -o /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/ -sid sub-1

/home/ch222974/Desktop/conda_envs/dce_env/bin/python3.7 /home/ch222199/Documents/GIT/mocoRec/dce_mri/slice_to_vol.py -p /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/pproc.json -d /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/sub-1/*/raw-rec/ -o /fileserver/fastscratch/cemre/VIDA_MRUs/MRU021622/sub-1/
