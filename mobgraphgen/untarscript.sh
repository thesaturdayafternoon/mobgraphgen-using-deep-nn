graphs_created=true
if [ "$graphs_created" = true ] ; then
    shopt -s nullglob
    dirs=( /content/drive/MyDrive/mobgraphgen/datasets/grid/min_dfscode_tensors/*/ )
    cnt=${#dirs[@]}
    echo $cnt
    mkdir /content/min_dfscode_tensors/
    for ((i=0;i<cnt;i++))
      do
        echo $i
        mkdir /content/min_dfscode_tensors/${i}
        tar -xf /content/drive/MyDrive/mobgraphgen/datasets/grid/min_dfscode_tensors/${i}.tar.bz2 -C /content/min_dfscode_tensors/${i}
    done
fi