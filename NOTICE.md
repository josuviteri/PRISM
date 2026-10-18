PRISM
Copyright (c) 2026 Josu Viteri and Gotzon Viteri.

This product includes software developed for the PRISM project.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. A copy of the License is
provided in the LICENSE file in the root of this repository, or you may obtain
it at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

================================================================================
Third-party software
================================================================================

This product depends on the following third-party software components. The
versions below are the direct dependencies declared in `requirements.txt`;
their transitive dependencies may have additional notices and license terms.

| Component | Version | License | License / notice |
|-----------|---------|---------|------------------|
| PyTorch | 2.14.1 | BSD 3-Clause | <https://github.com/pytorch/pytorch/blob/main/LICENSE> |
| torchvision | 0.29.1 | BSD 3-Clause | <https://github.com/pytorch/vision/blob/main/LICENSE> |
| NumPy | 2.1.3 | BSD 3-Clause | <https://github.com/numpy/numpy/blob/main/LICENSE.txt> |
| Pillow | 11.0.0 | HPND | <https://github.com/python-pillow/Pillow/blob/main/LICENSE> |
| PyYAML | 6.0.2 | MIT | <https://github.com/yaml/pyyaml/blob/main/LICENSE> |
| Weights & Biases (`wandb`) | 0.18.7 | MIT | <https://github.com/wandb/wandb/blob/main/LICENSE> |
| tqdm | 4.67.1 | MPL 2.0 / MIT | <https://github.com/tqdm/tqdm/blob/master/LICENCE> |
| py7zr | 0.22.0 | LGPL-2.1-or-later | <https://github.com/miurahr/py7zr/blob/master/LICENSE> |
| Matplotlib | 3.9.3 | Matplotlib License | <https://matplotlib.org/stable/project/license.html> |
| pytest | 8.3.4 | MIT | <https://github.com/pytest-dev/pytest/blob/main/LICENSE> |

Dataset attributions:

* **CIFAR-10** is obtained from the University of Toronto source described in
  the README and is distributed under the CIFAR-10 dataset terms (MIT).
* **MNIST** is obtained from the source described in the README and is
  distributed under the Creative Commons Attribution-ShareAlike 3.0 license
  (CC BY-SA 3.0).

The datasets are downloaded locally by `scripts/get_data.py` and are not
redistributed by this repository. Users must comply with the applicable
dataset terms when downloading or redistributing them.

