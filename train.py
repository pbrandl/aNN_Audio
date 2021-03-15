# Hyperparameter Settings
length_sample = 2048
batch_dim = 4
num_channels = 1

dataset = AudioDataset(preloader=preloader, device=device)
x, y = dataset.batchify(batch_dim=batch_dim, length_sample=length_sample, n_samples=5000)
x_train, x_valid, y_train, y_valid = dataset.split_data((x, y), 0.8, 0.2)

model = WaveNet(length_sample, max_dilation=11, num_hidden=1, num_blocks=1, device=device)
print("Train Data Shape: {}.".format(x.shape))
print("Receptive Field Length: {}.".format(model.length_rf))

logger = Logger(logger_path)

assert x_train.shape ==  y_train.shape and x_valid.shape == y_valid.shape, "Expected equal shape."

def train(model, x, y, loss_fn, optimizer):
    model.train()
    for x_batch, y_batch in zip(x, y):
        optimizer.zero_grad()
        prediction = model(x_batch)
        loss = loss_fn.forward(prediction, y_batch)
        loss.backward()
        optimizer.step()
        #print(loss.item())
    return loss.item()

@torch.no_grad()
def validate(model, x, y, loss_fn):
    loss_history = []
    model.eval()
    for x_sample, y_sample in zip(x, y):
        prediction = model(x_sample)
        loss_history.append(loss_fn.forward(prediction, y_sample))
    return sum(loss_history) / len(loss_history)

@torch.no_grad()
def pred_listening_test(model):
    print("now predicting listening_test")
    y_listening_test = model.predict_sequence(listening_test_drums)
    display(Audio(y_listening_test.cpu().numpy(), rate=44100))
    y_listening_test = model.predict_sequence(listening_test_fmix)
    display(Audio(y_listening_test.cpu().numpy(), rate=44100))

def fit(model, x_train, y_train, x_valid, y_valid, epochs, config, logger=None):
    assert x_train.shape == y_train.shape, "Expected data in equal shape."
    assert x_valid.shape == y_valid.shape, "Expected data in equal shape."
    lr = config['lr']

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    #loss_fn = nn.MSELoss(reduction='mean')
    #loss_fn = PerceptualLoss(scale=bark, samplerate=SR44100()).to(device)
    loss_fn = nn.L1Loss(reduction='mean')
    #loss_fn = cdpam.CDPAM()

    for epoch in range(int(epochs)):
        model.reset_previous_rf()
        loss_train = train(model, x_train, y_train, loss_fn, optimizer)
        loss_valid = validate(model, x_valid, y_valid, loss_fn)
        print("Epoch:", epoch, "\nAvg. valid. loss:", loss_valid.item())
        if logger is not None:
            logger.log('test', epoch, model, loss_valid)
    
        pred_listening_test(model)

    return loss_valid, loss_train

print(fit(model, x_train, y_train, x_valid, y_valid, epochs=50, config={'lr': 1e-4}, logger=None))
