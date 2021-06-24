# Slimmable Models

This directory contains the implmented Slimmable networks. Specifically, Slimmable ResNets, MobileNetV2, MobileNetV3, and EfficientNets.

We use a hirarchical representation to specify a slimmable network. More specifically, in slimmable networks, one has to carefully manipulate channel counts for each layer so that the neural network is executable (no input/output channel mismatches). To achieve this, we use a hirarchical design as follows:

    `SlimModel` -> `SlimStage` -> `SlimBlock` -> `SlimOps` (BN or Conv)

In this fashion, each level only has to care about the input/output channel mappings within that particular level to achieve a better flexibility and extensibility.

# Implement Your Own Slimmable Models

In case the network model you care about is not implemented already, we provide a guideline on how to implement a new slimmalbe network based on our code easily.

## Model architecture
In your new model, it would be much easier if you follow our hirarchical design. Specifically, we want to implement a new model `SlimModel` which consists of several `SlimStage` modules. Each `SlimStage` module consists of several `SlimBlock`, which is composed by `SlimOps`.

We have provided several `SlimBlock` modules such as `SlimBasicBlock` and `SlimBottleNeck` for ResNets, `SlimDWSConv` and `SlimInvertedResidual` for MobileNets and EfficientNets. They all reside in `dyn_model.py`. Note that the number of searchable layers for each block is implemented in the attribute `search_dim` for each of the module.

To implement your `SlimModel` module, it has to inherit `DynModel` from `dyn_model.py`. The necessary interface to implement is detailed in the next subsection. You can implement `SlimStem`, `SlimStage`, and `SlimHead` inside your `SlimModel`. For many architectures, `SlimStem` and `SlimHead` are just `SlimConv2d`. That is, replace your first and last layer with `SlimConv2d` and implement your own `SlimStage` (feel free to check `slim_mbv2.py`, `slim_mbv3.py`, and `slim_resnet.py` for references).

## API

To aid understanding, let's assume we want to build a model `SlimModel` with a stem using `SlimConv2d`, a head using `SlimConv2d` and has three stages, where each stage have 3 `SlimInvertedResidual` blocks.

The new class has to implement the interface of `DynModel` in `dyn_model.py` by inheriting `DynModel` and implement the following functions and attributes:

### `search_dim`

`self.search_dim` for `DynModel` is the number of optimizable variables (the number of layers). This is directly used by the external optimizer to generate architecture. An example would be `np.random.rand(self.search_dim)` for random search. Note that the layers that share the same channel counts either due to depth-wise separable convolution or skip connection are already fused. More specifically, let's say four layers share the same channel counts, it will just be one dimension in `self.search_dim`. This designs makes sure that the optimizers that operate on this representation is efficient without trying to search for redundant dimensions.

To implement a new model, the designer is responsible for counting the dimensions correctly without having redundant dimensions. In our example, we have 1 dimension for the stem, each stage has (1+3) dimensions, and since the head has fixed output classes, it is not tunable in the output dimension and its input dimension depends on the last stage. Let's look into the (1+3) dimensions. Since each `SlimInvertedResidual` has 2 searchable dimension according to its `search_dim`, we have in total 6 flexible dimensions. However, within each stage, there is residual connections connecting the last layer of each `SlimInvertedResidual` block. As a result, we have 1 dimension controlling the residual channel counts and 3 other dimensions for the rest. Overall, this toy model would have 13 searchable dimensions.


### `decode_wm(self, wm)`
This is a member function that converts a list of layer-wise width multipliers `wm` (values between 0 and 1), to real channel counts. Moreover, `wm` is a flat list while the output will be hirarchical: list of lists (#stages of lists). In our implementation, we introduce an attribute called `channel_space` for each of the model, which contains the original channel counts for each of the variable in `wm`. Moreover, `channel_space` is hirarchical (list of lists). The output of this function will be a hirarical representation of channel counts.

In our example, we have 13 dimensions for `wm`. The hirarchical representation of the width multipliers would be [$w_1$, [$w_2,w_3,w_4,w_5$], [$w_6,w_7,w_8,w_9$], [$w_{10},w_{11},w_{12},w_{13}$]]. The output of this function will be channel counts as opposed to channel multipliers: [$c_1$, [$c_2,c_3,c_4,c_5$], [$c_6,c_7,c_8,c_9$], [$c_{10},c_{11},c_{12},c_{13}$]].

### `decode_ch(self, ch)`
This is a function that maps the hirarchical representation of channel counts (output from `self.decode_wm`) to real channel counts. The difference between the two is that in the real channel counts, some layers have to share the same channel counts. Those layers are represented using a single variable in `ch`. As a result, this function expand `ch` to a have a channel count for each of layer.

In our example, we share the channel counts for the last Convolutional layers in each of the block within a stage. We are going to expand them with this function. The input `ch` would be [$c_1$, [$c_2,c_3,c_4,c_5$], [$c_6,c_7,c_8,c_9$], [$c_{10},c_{11},c_{12},c_{13}$]] and the output would be [$c_1$, [$c_3,c_2,c_4,c_2,c_5,c_2$], [$c_7,c_6,c_8,c_6,c_9,c_6$], [$c_{11},c_{10},c_{12},c_{10},c_{13},c_{10$]]

### `set_real_ch(self, real_ch)`
Finally, this simply applies recursively the output channel counts obtained from `self.decode_ch` to the `SlimStage` modules. Note that this function mainly uses `m.set_out_ch`, `m.set_in_ch`, and `m.set_groups` for slimmable a module `m`. It is also in charge of setting the input channels for the last layer.

In our example, `real_ch` would be [$c_1$, [$c_3,c_2,c_4,c_2,c_5,c_2$], [$c_7,c_6,c_8,c_6,c_9,c_6$], [$c_{11},c_{10},c_{12},c_{10},c_{13},c_{10$]]. This function set the channel configurations with the following code snippet:

    stem.set_out_ch(real_ch[0])

    stage1.set_in_ch(real_ch[0])
    stage1.set_out_ch(real_ch[1])

    stage2.set_in_ch(real_ch[1][-1])
    stage2.set_out_ch(real_ch[2])

    stage3.set_in_ch(real_ch[2][-1])
    stage3.set_out_ch(real_ch[3])

    head.set_in_ch(real_ch[3][-1])