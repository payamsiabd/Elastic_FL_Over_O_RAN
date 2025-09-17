from core.utils import *
from core.neural_networks import *


def evaluate(model):
    model.eval(); correct = total = 0
    with torch.inference_mode():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

def local_train(global_model, loader):
    m = copy.deepcopy(global_model).to(DEVICE)
    m.train()
    opt = optim.SGD(m.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(m(xb), yb)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

def fedavg_inplace(global_model, states, weights):
    total = float(sum(weights))
    base = {k: torch.zeros_like(v) for k, v in states[0].items()}
    for st, w in zip(states, weights):
        a = w / total
        for k in base:
            base[k].add_(st[k], alpha=a)
    global_model.load_state_dict(base)


# ------------------ config ------------------
ALPHA = 5
LOCAL_EPOCHS = 1
LOCAL_BS = 64
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 45
torch.manual_seed(SEED); np.random.seed(SEED)
num_services = 3
num_global_rounds = 20
num_users = [50,50,50]


success_status = load_pickle("success_status.pkl")
success_status = success_status['success_status']
for gr in range(num_global_rounds):
    print(f"+++++++++++++++++++++++++++++++++GLOBAL ROUND {gr}+++++++++++++++++++++++++++++++++")
    for s in range(num_services):
        print(f"--------------------------------Service {s}-------------------------------------")
        print(f"success status of service {s} = {success_status[gr, s]}")
        print(f"total successful users of service {s}: {np.bincount(success_status[gr, s])[1]}")
        print("--------------------------------------------------------------------------------")




# Your status vector (example from your message)
# status = np.array([
#  True, True, True, True, True, True, True,  True, True, True,
#   True,  True, True,  True, True, True, True, True,  True, True,
#  True, True, True,  True,  True,  True, True, True, True, True,
#  True,  True, True, True, True,  True, True, True, True, True,
#  True,  True,  True,  True,  True, True,  True, True, True,  True
# ], dtype=bool)

# ------------------ per-service data/model ------------------
for s in range(num_services):
    # pick dataset + transform + model per service
    if s == 0:
        # CIFAR-10 (3×32×32)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
        ])
        trainset = torchvision.datasets.CIFAR10("../data", train=True, download=True, transform=transform)
        testset  = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=transform)
        global_model = CIFAR10_CNN().to(DEVICE)
        ds_name = "CIFAR-10"
    elif s == 1:
        # Fashion-MNIST (1×28×28)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.2860,), (0.3530,))
        ])
        trainset = torchvision.datasets.FashionMNIST("../data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST("../data", train=True, download=True, transform=transform)
        global_model = FMNIST_CNN().to(DEVICE)
        ds_name = "Fashion-MNIST"
    elif s == 2:
        # MNIST (1×28×28)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST("../data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST("../data", train=False, download=True, transform=transform)
        global_model = MNIST_CNN().to(DEVICE)
        ds_name = "MNIST"

    test_loader = DataLoader(testset, batch_size=256, shuffle=False)


    print(f"++++++++++++++++++++++++ FL Training of Service {s} ({ds_name}) +++++++++++++++++++++++++")
    client_indices = dirichlet_split_strict(trainset.targets, num_users[s], ALPHA)
    # num_classes = 10
    # _ = plot_dirichlet_allocation(client_indices, trainset.targets, num_classes,
    #                               title=f"{ds_name} | Dirichlet split (alpha={ALPHA}, clients={num_users[s]})")
    # plt.show()

    client_loaders, client_sizes = [], []
    for idxs in client_indices:
        ds = Subset(trainset, idxs.tolist())
        client_loaders.append(DataLoader(ds, batch_size=LOCAL_BS, shuffle=True))
        client_sizes.append(len(idxs))
    client_sizes = np.array(client_sizes, dtype=int)

    # ------------------ FL loop ------------------
    for rnd in range(num_global_rounds):
        status = success_status[rnd, s]
        selected = [i for i in range(num_users[s]) if status[i]]
        if not selected:
            print(f"[Round {rnd}] No successful clients; skipping.")
            continue

        client_states, weights = [], []
        for i in selected:
            st = local_train(global_model, client_loaders[i])
            client_states.append(st)
            weights.append(int(client_sizes[i]))  # weight = #datapoints

        fedavg_inplace(global_model, client_states, weights)
        acc = evaluate(global_model)
        print(f"[Round {rnd}] aggregated {len(selected)}/{num_users[s]} clients | test acc = {acc:.2%}")

    print("Done.")