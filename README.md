# CUSTOM EFFICIENT NEURAL NETWORK FOR HISTOPATHOLOGICAL LUNG CANCER CLASSIFICATION
This paper explores the development and evaluation of a small custom image classifier for lung cancer histopathological images. Leveraging Deep Neural Networks, the model should aid enhance medical diagnosis efficiency. To optimize deployment, the architecture focuses on minimizing model size without resorting to post-development compression techniques. The algorithm demonstrates promising results, achieving a 91.89% reduction in parameter size compared to EfficientNet variant B0, with a deployable model parameter size of approximately 1.6 MegaBytes (MBs). Training results reveal superior performance, with a final accuracy of 99% and a loss of 0.018140. Despite longer training times compared to some variants, the model's effectiveness underscores the potential of tailored deep learning solutions in medical diagnostics, albeit with hardware-dependent considerations.

The hosted web applications can be found here. https://lungcancerhistopathy.streamlit.app/

Note: To use notebook and trained weights, rename `model_v4_0199.pt` to `model.pt` in checkpoints folder.
