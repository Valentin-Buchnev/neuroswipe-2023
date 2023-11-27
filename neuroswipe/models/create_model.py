from neuroswipe.models import Model


def create_model(arch):
    if arch == "model":
        return Model()
    else:
        raise Exception("Unknown architechure {}".format(arch))
