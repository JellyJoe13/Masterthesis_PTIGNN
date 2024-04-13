echo "starting script..."

call conda activate Masterarbeit_PTGNN

pause

call python full_test_script.py "configs/final_test/bace/instructions_bace_ptree_edge_default.yaml" --verbose

call python full_test_script.py "configs/final_test/bace/instructions_bace_ptree_edge_selective_rwse.yaml" --verbose

call python full_test_script.py "configs/final_test/bace/instructions_bace_ptree_vertex_default_rwse.yaml" --verbose

call python full_test_script.py "configs/final_test/bace/instructions_bace_ptree_vertex_selective_rwse.yaml" --verbose

pause

call conda deactivate