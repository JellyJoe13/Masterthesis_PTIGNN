echo "starting script..."

call conda activate Masterarbeit_PTGNN

pause

call python hyper_opt_script_v3.py "configs/benchmarking/unmasked/rs_unmasked_dummy.yaml" --verbose
call python full_test_script.py "configs/final_test/unmasked/rs_unmasked_dummy.yaml" --verbose

pause

echo "stop here"
pause

:: ba
call python hyper_opt_script_v3.py "configs\benchmarking\bindingaffinity\benchmark_instructions_bindingaffinity_ptree_selective_rwse_pelu.yaml" --verbose -gpu 0.3

call python full_test_script.py "configs\final_test\bindingaffinity\instructions_bindingaffinity_ptree_selective_rwse_pelu.yaml" --verbose

pause

call conda deactivate