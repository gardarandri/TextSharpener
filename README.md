
TextSharpener
=============

This is the code behind the following post: <https://gardarandri.github.io/TextSharpener/>.

Usage
=====

Currently the code needs cleaning but can be tinkered with.

* model/model.py is used for training and inference.
* generator/GenImages.py is used to generate training data.
* savedmodels contains a pretrained model.

Examples
========

The following are examples from the validation set after training.

![](docs/vallabel.png)
![](docs/valtrain.png)
![](docs/valoutput.png)

![](docs/vallabel2.png)
![](docs/valtrain2.png)
![](docs/valoutput2.png)

![](docs/vallabel3.png)
![](docs/valtrain3.png)
![](docs/valoutput3.png)

![](docs/vallabel4.png)
![](docs/valtrain4.png)
![](docs/valoutput4.png)

Left: The original image, Middle: The blurred image, Right: The outputted image after training.
