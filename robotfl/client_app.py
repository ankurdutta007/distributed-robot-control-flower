"""A Flower client = ONE robot cell. It trains on its own sensor data and only returns weights."""
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from robotfl.task import make_station_data, train_epochs, mse

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    station_id = context.node_config["partition-id"]
    params = msg.content["arrays"].to_numpy_ndarrays()
    lr = msg.content["config"]["lr"]
    epochs = int(context.run_config["local-epochs"])

    X, Y = make_station_data(station_id)
    rnd = int(msg.content["config"].get("server-round", 0))
    params, loss = train_epochs(params, X, Y, epochs=epochs, lr=lr, seed=rnd * 100 + station_id)

    reply = RecordDict({
        "arrays": ArrayRecord(params),
        "metrics": MetricRecord({"train_loss": loss, "num-examples": len(X)}),
    })
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    station_id = context.node_config["partition-id"]
    params = msg.content["arrays"].to_numpy_ndarrays()
    X, Y = make_station_data(station_id, n=1000, seed=7)
    reply = RecordDict({"metrics": MetricRecord({"eval_loss": mse(params, X, Y), "num-examples": len(X)})})
    return Message(content=reply, reply_to=msg)
