"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fltabular.task import (
    IncomeClassifier,
    get_weights,
    load_data,
    set_weights,
    train_and_evaluate,
    load_pretrained_model,
)


class Net(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, y_test, input_dim):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.input_dim = input_dim
        self.y_test = y_test
          

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        #train(self.net, self.trainloader)
        #rocy(self.net, self.trainloader, self.testloader)
        model = load_pretrained_model(self.input_dim)
        train_and_evaluate(model, self.trainloader, self.testloader, self.y_test, num_epochs=2)
        #train_and_evaluate(model, train_loader, test_loader, y_test, num_epochs=5)
        return get_weights(self.net), len(self.trainloader), {}

   # def evaluate(self, parameters, config):
   #     set_weights(self.net, parameters)
   #     loss, accuracy = evaluate(self.net, self.testloader)
   #     return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]

    train_loader, test_loader, y_test, input_dim = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    
    net = IncomeClassifier()
    return FlowerClient(net, train_loader, test_loader, y_test, input_dim).to_client()
    
    #net = Net(input_dim=42)
    #return Net(net, train_loader, test_loader, input_dim ).to_client()


app = ClientApp(client_fn=client_fn)


