def create_model(use_pretrained: bool, feature_extract: bool):
    model = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)

    # Reshape the final layer.
    fc_n_features = model.fc.in_features
    model.fc = nn.Linear(fc_n_features, 5)
    model.conv1 = Conv2d(
        9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    return model
