PYTHON=$(which python3)
conda activate py37
cd F:/resource_provision_mec/codes/min_max || exit

num_group_per_instance=10
simulation_times=30

for((simulation_no=1;simulation_no<="$simulation_times";simulation_no++))
do
   # PYTHON collect_pulp_solutions.py "$simulation_no" "$num_group_per_instance" >> "result/PULP_solutions/shell_output_$simulation_no".txt &
   PYTHON collect_pulp_solutions.py "$simulation_no" "$num_group_per_instance" &
done

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"