PYTHON=$(which python3)
conda activate py37
cd F:/resource_provision_mec/codes || exit
for i in {1..20}
do
   PYTHON main.py "$i" >> "shell_output_$i".txt &
done

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"