# ARG BASE_IMAGE=jetbot-base:jp44
ARG BASE_IMAGE=${BASE_IMAGE}
FROM ${BASE_IMAGE}

RUN python3 -c "import torchvision; \
                model = torchvision.models.alexnet(pretrained=True); \
                model = torchvision.models.squeezenet1_1(pretrained=True); \
                model = torchvision.models.resnet18(pretrained=True); \
                model = torchvision.models.resnet34(pretrained=True)"
