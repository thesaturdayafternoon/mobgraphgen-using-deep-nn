shopt -s nullglob
dirs=( /content/drive/MyDrive/mobgraphgen/datasets/grid/min_dfscode_tensors/*/ )
echo "There are ${#dirs[@]} (non-hidden) directories"

cnt=${#dirs[@]}
echo $cnt

for ((i=0;i<cnt;i++));
  do
    sleep 2
    echo $i
    cd /content/drive/MyDrive/mobgraphgen/datasets/grid/min_dfscode_tensors && tar cfj "${i}.tar.bz2" -C ${i} .
done