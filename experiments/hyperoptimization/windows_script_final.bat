echo "starting script..."

call conda activate Masterarbeit_PTGNN

pause

call python hyper_opt_script_v3.py "configs/benchmarking/tox21/benchmark_instructions_tox21_ptree_vertex_default_e_linred.yaml" --verbose -device cpu -cpu 5 -gpu 0

call python full_test_script.py "configs/final_test/tox21/instructions_tox21_ptree_vertex_default_e_linred.yaml" --verbose -device cpu -cpu 5 -gpu 0

pause

call python hyper_opt_script_v3.py "configs/benchmarking/rs/benchmark_instructions_rs_ptree_vertex_default_e_linred.yaml" --verbose -device cpu -cpu 5 -gpu 0

call python full_test_script.py "configs/final_test/rs/instructions_rs_ptree_vertex_default_e_linred.yaml" --verbose -device cpu -cpu 8 -gpu 0

call conda deactivate