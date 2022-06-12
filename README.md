# GOOD
GOOD: A Graph Out-of-Distribution Benchmark

------------------------------

[license-url]: https://github.com/divelab/GOOD/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg

[![Documentation Status](https://readthedocs.org/projects/good/badge/?version=latest)](https://good.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![License][license-image]][license-url]
> We are actively building the document

* [Overview](overview)
* [Why GOOD?](why-good?)

## Overview

------------------------------------------------------
Out-of-distribution (OOD) learning deals with scenarios in which training and test data follow different distributions. 
Although general OOD problems have been intensively studied in machine learning, graph OOD is only an emerging area of research. 
Currently, there lacks a systematic benchmark tailored to graph OOD method evaluation. 
This project is for Graph Out-of-distribution development, known as GOOD.
We explicitly make distinctions between covariate and concept shifts and design data splits that accurately reflect different shifts. 
We consider both graph and node prediction tasks as there are key differences when designing shifts. 
Currently, GOOD contains 8 datasets with 14 domain selections. When combined with covariate, concept, and no shifts, we obtain 42 different splits. 
We provide performance results on 7 commonly used baseline methods with 10 random runs. 
This results in 294 dataset-model combinations in total. Our results show significant performance gaps between in-distribution and OOD settings. 
We hope our results also shed light on different performance trends between covariate and concept shifts by different methods. 
This GOOD benchmark is a growing project and expects to expand in both quantity and variety of resources as the area develops.
Any contribution is welcomed!

## Why GOOD?

----------------------------------------------------
Whether you are an experienced researcher for graph out-of-distribution problem or a first-time learner of graph deep learning, 
here are several reasons for you to use GOOD as your Graph OOD research, study, and development toolkit.

* **Easy-to-use APIs:**
* **Flexibility:**
* **Easy-to-extend architecture:**
* **Easy comparisons with the leaderboard:**
