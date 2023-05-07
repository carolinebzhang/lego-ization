from models import CycleGAN
import torch
from datasets import get_data

images_train_loader, images_test_loader = get_data(1)
def train(epochs=50, save=True, load=False, model_path='model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CycleGAN()
    if load:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    path = "model"
    path_num = 0
    path_ending = ".pth"
    # paper uses lr=.0002, batch size=1, 100 epochs with lr and then 100 more with decaying lr
    # 128x128 or 256x256 images
    opt_G_A2B = torch.optim.Adam(model.G_A2B.parameters(), lr=.0002)
    opt_G_B2A = torch.optim.Adam(model.G_B2A.parameters(), lr=.0002)
    opt_D_A = torch.optim.Adam(model.D_A.parameters(), lr=.0002)
    opt_D_B = torch.optim.Adam(model.D_B.parameters(), lr=.0002)

    for epoch in range(epochs):
        model.train()
        print(f"epoch:{epoch}")
        #for i, batch in batches: # TODO FIGURE OUR DATA LOADING / BATCHING
        for i, data in enumerate(images_train_loader):
            real_a, real_b = data['lego_image'], data['real_image']
            if device == 'cuda':
                 real_a = real_a.cuda()
                 real_b = real_b.cuda()
            opt_G_A2B.zero_grad()
            opt_G_B2A.zero_grad()
            opt_D_A.zero_grad()
            opt_D_B.zero_grad()

            cycle_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss = model(real_a, real_b)
            
            g_A2B_loss.backward(retain_graph=True)
            g_B2A_loss.backward(retain_graph=True)

            opt_G_A2B.step()
            opt_G_B2A.step()

            d_A_loss.backward(retain_graph=True)
            d_B_loss.backward()

            opt_D_A.step()
            opt_D_B.step()

        total_fake_A_acc = 0
        total_fake_B_acc = 0
        total_real_A_acc = 0
        total_real_B_acc = 0
        n = 0
        for i, data in enumerate(images_test_loader):
            real_a, real_b = data['lego_image'], data['real_image']
            if device == 'cuda':
                real_a = real_a.cuda()
                real_b = real_b.cuda()

            fake_A_acc, fake_B_acc, real_A_acc, real_B_acc = model.test(real_a, real_b)

            total_fake_A_acc += fake_A_acc
            total_fake_B_acc += fake_B_acc
            total_real_A_acc += real_A_acc
            total_real_B_acc += real_B_acc
            n += 1

        print(f"fake_A_acc:{total_fake_A_acc/n} fake_B_acc:{total_fake_B_acc/n} real_A_acc:{total_real_A_acc/n} real_B_acc:{total_real_B_acc/n}")
        model_path_new = path + str(path_num) + path_ending

        if save:
            torch.save(model.state_dict(), model_path_new)

        path_num+=1



    if save:
            torch.save(model.state_dict(), model_path_new)

if __name__=="__main__":
    train()