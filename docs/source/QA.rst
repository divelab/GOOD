Q&A
====

* How can one add a new algorithm and benchmark it with GOOD?

`Check this issue <https://github.com/divelab/GOOD/issues/4#issuecomment-1228953087>`_.

* Can I use my own dataloader?

Yes. In `GOOD/data/good_loaders/` folder, create a new python file and inherit the basic loader class in `BaseLoader.py`. Don't forget to register it.

* Can I use more flexible pipeline structure?

Yes. In `GOOD/kernel/pipelines/` folder, create a new python file or copy the `basic_pipeline.py`, then build or modify your own pipeline! It is much easier to copy and
modify the original pipeline than building a new one from scratch unless you are completely familiar with this project.

For more questions, please submit issues or open discussions in our GitHub repo.