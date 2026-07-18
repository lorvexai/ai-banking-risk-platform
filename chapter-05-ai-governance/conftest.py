# tests/conftest.py
# Prevent pytest from collecting module-level test_* factory functions
# from model_governance.validation_framework — those are validation test
# builders, not pytest test cases.

collect_ignore_glob = []
