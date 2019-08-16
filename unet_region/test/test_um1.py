import torch

start = 0
end = 29

snake_u = torch.arange(start, end).unsqueeze(1)
snake_u = torch.cat((snake_u, torch.tensor([[start]])))

snake_um1 = torch.cat((snake_u[-2:, :],
                        snake_u[0:-2, :]), 0)
snake_up1 = torch.cat((snake_u[1:, :],
                        snake_u[0, :].unsqueeze(0)), 0)

print(snake_u)
print(snake_um1)
print(snake_up1)
