"""
ui 模块
包含所有 UI 组件
"""

from .sidebar import render_sidebar
from .dialogs import (
    show_basic_cases_dialog,
    edit_basic_case_dialog,
    show_supp_cases_dialog,
    edit_supp_case_dialog,
    show_prompt_dialog,
    show_tea_examples_dialog,
    manage_tea_examples_dialog,
    edit_tea_example_dialog
)
from .tab1_interactive import render_tab1
from .tab2_batch import render_tab2
from .tab3_knowledge import render_tab3
from .tab4_cases import render_tab4
from .tab5_finetune import render_tab5
from .tab6_prompts import render_tab6

__all__ = [
    'render_sidebar',
    'show_basic_cases_dialog',
    'edit_basic_case_dialog',
    'show_supp_cases_dialog',
    'edit_supp_case_dialog',
    'show_prompt_dialog',
    'show_tea_examples_dialog',
    'manage_tea_examples_dialog',
    'edit_tea_example_dialog',
    'render_tab1',
    'render_tab2',
    'render_tab3',
    'render_tab4',
    'render_tab5',
    'render_tab6'
]