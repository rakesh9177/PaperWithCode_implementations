import torch
import torch.nn as nn

def Adam(model_parameters, loss_function, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_t = [torch.zeros_like(param) for param in model_parameters]
    print(m_t)
    v_t = [torch.zeros_like(param) for param in model_parameters]
    t = 0
    
    for t in range(1, 100):
        loss = loss_function()  # Compute the loss
        loss.backward()  # Compute gradients
        
        for i, p in enumerate(model_parameters):
            g_t = p.grad
            m_t[i] = beta1 * m_t[i] + (1 - beta1) * g_t
            v_t[i] = beta2 * v_t[i] + (1 - beta2) * (g_t ** 2)
            
            m_t_hat = m_t[i] / (1 - beta1 ** t)
            v_t_hat = v_t[i] / (1 - beta2 ** t)
            
            p.data -= lr * m_t_hat / (torch.sqrt(v_t_hat) + epsilon)
            
        # Clear gradients for the next iteration
        for p in model_parameters:
            p.grad.zero_()
    
    return model_parameters

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
y = torch.tensor([[2.0], [4.0], [6.0]], requires_grad=False)

# Instantiate the model and define the loss function
model = SimpleModel()
criterion = nn.MSELoss()

model_parameters = model.parameters()
Adam(model_parameters, lambda: criterion(model(x), y))  # lambda function for loss computation

# Print the updated model parameters
for param in model.parameters():
    print(param.data)




