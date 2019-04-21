# from scripts.run_rbf_comparison import run_rbf_comparison
# from scripts.test_inv_hvp import test_inv_hvp
from scripts.run_adversarial_atk_svm import run_adversarial_atk_svm
# from scripts.run_data_poisoning_multiple import run_data_poisoning_multiple

# run_rbf_comparison()
# test_inv_hvp()
# run_adversarial_atk_svm(weight_decay=0.001, num_iter=100, attack_target="grad")
# run_adversarial_atk_svm(weight_decay=0.0002, num_iter=100, attack_target="grad")
run_adversarial_atk_svm(weight_decay=0.001, num_iter=100, attack_target="sv")
# run_adversarial_atk_svm(weight_decay=0.0002, num_iter=100, attack_target="sv")
# run_data_poisoning_multiple()
