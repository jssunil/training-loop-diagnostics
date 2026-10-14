"""Build notebooks/training_loop_diagnostics.ipynb. Execute it with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/training_loop_diagnostics.ipynb
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Training Loop Diagnostics\n"
       "Take a small model (nanoGPT) and a real loop, and make it tell the truth about itself.\n\n"
       "Each part calls one script in `src/`. The numbers and plots below are the outputs of this run. "
       "See `README.md` for the written answers and `PLAN.md` for the method."),
    code("# Setup. On Colab this clones the repository; locally it only moves to the repository root.\n"
         "import os, sys, pathlib\n"
         "if 'google.colab' in sys.modules and not pathlib.Path('training-loop-diagnostics').exists():\n"
         "    !git clone -q https://github.com/jssunil/training-loop-diagnostics\n"
         "    %pip install -q -r training-loop-diagnostics/requirements.txt\n"
         "root = pathlib.Path('training-loop-diagnostics') if pathlib.Path('training-loop-diagnostics').exists() else pathlib.Path.cwd()\n"
         "if root.name == 'notebooks':\n"
         "    root = root.parent\n"
         "os.chdir(root); sys.path.insert(0, str(root.resolve()))\n"
         "import torch\n"
         "from src.device import pick_device, env_info\n"
         "print(env_info(pick_device()))"),
    md("## Part 1: Every tensor shape in one step"),
    code("from src import p1_shapes\n_ = p1_shapes.run()"),
    md("## Part 2: Verify one gradient by hand"),
    code("from src import p2_gradcheck\n_ = p2_gradcheck.run()"),
    md("## Part 3: Break gradient accumulation on purpose"),
    code("from src import p3_accum_bug\nfrom IPython.display import Image, display\n"
         "_ = p3_accum_bug.run()\ndisplay(Image('results/p3_accum_bug.png'))"),
    md("## Part 4: The grad norm moves before the loss"),
    code("from src import p4_gradnorm\n_ = p4_gradnorm.run()\ndisplay(Image('results/p4_gradnorm.png'))"),
    md("## Part 5: MFU"),
    code("from src import p5_mfu\n_ = p5_mfu.run()"),
    md("## Part 6: 0.1 in fp32, bf16 and fp8 E4M3"),
    code("from src import p6_bits\n_ = p6_bits.run()"),
    md("## Tests"),
    code("!{sys.executable} -m pytest -q tests"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "training_loop_diagnostics.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
