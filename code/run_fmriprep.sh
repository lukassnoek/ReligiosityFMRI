bids_dir=`realpath $PWD`
out_dir=$bids_dir/derivatives

fmriprep-docker $bids_dir $out_dir \
        --image poldracklab/fmriprep:1.0.15 \
        --nthreads 2 \
        --omp-nthreads 3 \
        --ignore slicetiming \
        --output-space template \
        --use-syn-sdc \
        --fs-license-file /usr/local/freesurfer/license.txt \
