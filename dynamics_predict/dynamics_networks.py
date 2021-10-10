import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsNetwork(nn.Module):
    """ Forward dynamics prediction network: (s, a, params) -> (s_) """
    def __init__(self, state_space, action_space, param_dim, hidden_dim=256, hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr=1e-3, gamma=0.99):
        super(DynamicsNetwork, self).__init__()
        if isinstance(state_space, int): # pass in state_dim rather than state_space
            self._state_dim = state_space
        else:
            self._state_space = state_space
            self._state_shape = state_space.shape
            if len(self._state_shape) == 1:
                self._state_dim = self._state_shape[0]
            else:  # high-dim state
                raise NotImplementedError 

        try:
            self._action_dim = action_space.n
        except:
            self._action_dim = action_space.shape[0]
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self._param_dim = param_dim
        self.num_hidden_layers = num_hidden_layers
        self.input_layer =  nn.Linear(self._state_dim+self._action_dim+self._param_dim, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        self.hidden_layers = nn.ModuleList(self.hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.output_layer =  nn.Linear(hidden_dim, self._state_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x=self.hidden_activation(self.input_layer(x))
        for hl in self.hidden_layers:
            x=self.hidden_activation(hl(x))
        x=self.output_layer(x)
        if self.output_activation is not None:
            x=self.output_activation(x)
        return x

    # batch = [[[s, a, params], s_],  ...]
    def saparm_to_inputs(self, batch, device):
        return torch.from_numpy(np.stack([np.concatenate([x.astype(np.float32) for x in data]) for data in batch],
                                         axis=0)).to(device)


class SINetwork(nn.Module):
    """ System identification network: (s, a, s, a, ..., s, a) -> (param) """
    def __init__(self, state_space, action_space, param_dim, frame_stack=5, hidden_dim=256,\
         hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr=1e-3, gamma=0.99):
        super(SINetwork, self).__init__()
        if isinstance(state_space, int): # pass in state_dim rather than state_space
            self._state_dim = state_space
        else:
            self._state_space = state_space
            self._state_shape = state_space.shape
            if len(self._state_shape) == 1:
                self._state_dim = self._state_shape[0]
            else:  # high-dim state
                raise NotImplementedError 

        try:
            self._action_dim = action_space.n
        except:
            self._action_dim = action_space.shape[0]
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self._param_dim = param_dim
        self.num_hidden_layers = num_hidden_layers
        self.input_layer =  nn.Linear(frame_stack*(self._state_dim+self._action_dim), hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        self.hidden_layers = nn.ModuleList(self.hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.output_layer =  nn.Linear(hidden_dim, self._param_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x=self.hidden_activation(self.input_layer(x))
        for hl in self.hidden_layers:
            x=self.hidden_activation(hl(x))
        x=self.output_layer(x)
        if self.output_activation is not None:
            x=self.output_activation(x)
        return x

class DynamicsParamsOptimizer():
    """ 
    Dynamics parameters optimization model (gradient-based) based on a trained 
    forward dynamics prediction network: (s, a, learnable_params) -> s_ with real-world data. 
    """
    def __init__(self, state_space, action_space, param_dim, param_ini_v, hidden_dim=256, hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr=1e-2, gamma=0.99):
        self.params = torch.tensor(param_ini_v, dtype=torch.float, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.params], lr=lr)
        self.dynamics_model = DynamicsNetwork(state_space, action_space, param_dim, hidden_dim, hidden_activation, output_activation, num_hidden_layers, lr, gamma)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """ s,a concat with param (learnable) -> s_ """
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

class DynamicsEncoder(nn.Module):
    """ Dynamics parameters encoding network: (params) -> (latent code) """
    def __init__(self, param_dim, latent_dim, hidden_dim=256, hidden_activation=F.relu, output_activation=F.tanh, num_hidden_layers=4, lr=1e-3, gamma=0.99):
        super(DynamicsEncoder, self).__init__()
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self._param_dim = param_dim
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer =  nn.Linear(self._param_dim, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        self.hidden_layers = nn.ModuleList(self.hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.output_layer =  nn.Linear(hidden_dim, latent_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x=self.hidden_activation(self.input_layer(x))
        for hl in self.hidden_layers:
            x=self.hidden_activation(hl(x))
        x=self.output_layer(x)
        if self.output_activation is not None:
            x=self.output_activation(x)
        return x

class DynamicsVariationalEncoder(DynamicsEncoder):
    """ Dynamics parameters encoding network: (params) -> (latent mu, latent sigma) """
    def __init__(self, param_dim, latent_dim, hidden_dim=256, hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr=1e-3, gamma=0.99):
        super().__init__(param_dim, latent_dim, hidden_dim, hidden_activation, output_activation, num_hidden_layers, lr, gamma)
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self._param_dim = param_dim
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers
        self.logvar_limit = 2.

        self.output_mu =  nn.Linear(hidden_dim, latent_dim)
        self.output_logvar = nn.Linear(hidden_dim, latent_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x=self.hidden_activation(self.input_layer(x))
        for hl in self.hidden_layers:
            x=self.hidden_activation(hl(x))
        mu=self.output_mu(x)
        logvar=self.output_logvar(x)
        if self.output_activation is not None:
            mu=self.output_activation(mu)  
        logvar = torch.clamp(logvar, -self.logvar_limit, self.logvar_limit)

        return mu, logvar


class DynamicsDecoder(nn.Module):
    """ Dynamics parameters decoding network: (latent code) -> (params) """
    def __init__(self, param_dim, latent_dim, hidden_dim=256, hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr=1e-3, gamma=0.99):
        super(DynamicsDecoder, self).__init__()
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self._param_dim = param_dim
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer =  nn.Linear(self.latent_dim, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        self.hidden_layers = nn.ModuleList(self.hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.output_layer =  nn.Linear(hidden_dim, self._param_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x=self.hidden_activation(self.input_layer(x))
        for hl in self.hidden_layers:
            x=self.hidden_activation(hl(x))
        x=self.output_layer(x)
        if self.output_activation is not None:
            x=self.output_activation(x)
        return x


class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, encoder_hidden_layers=2, decoder_hidden_layers=2, hidden_activation=F.relu, latent_dim=2, lr=1e-3):
        super(VAE, self).__init__()
        self.hidden_activation=hidden_activation

        # encoder part
        self.enc_input_layer = nn.Linear(x_dim, hidden_dim)
        self.enc_hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(encoder_hidden_layers)]
        self.enc_hidden_layers = nn.ModuleList(self.enc_hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.enc_output_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_output_std = nn.Linear(hidden_dim, latent_dim)

        # decoder part
        self.dec_input_layer = nn.Linear(latent_dim, hidden_dim)
        self.dec_hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(decoder_hidden_layers)]
        self.dec_hidden_layers = nn.ModuleList(self.dec_hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.dec_output_layer = nn.Linear(hidden_dim, x_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def encode(self, x):
        h=self.enc_input_layer(x)
        for ehl in self.enc_hidden_layers:
            h=self.hidden_activation(ehl(h))
        mu = self.enc_output_mu(h)
        logvar = self.enc_output_std(h)
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps*std+mu
        return z
        
    def decode(self, z):
        h=self.dec_input_layer(z)
        for dhl in self.dec_hidden_layers:
            h=self.hidden_activation(dhl(h))
        h = self.dec_output_mu(h)
        out = F.sigmoid(h) 
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def loss_function_VAE(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

class EmbeddingDynamicsNetwork(nn.Module):
    """ Common class for dyanmics prediction network with dynamics embedding as input: (s,a, alpha) -> s' """
    def __init__(self, ):
        super(EmbeddingDynamicsNetwork, self).__init__()
        self.dynamics_model = None
        self.ender = None
        self.decoder = None
        self.optimizer1 = None
        self.optimizer2 = None
        self.loss_dynamics = nn.MSELoss()
        self.loss_recon = nn.MSELoss()

    def forward(self, sa, theta):
        pass

    def save_model(self, path):
        if self.dynamics_model:
            torch.save(self.dynamics_model.state_dict(), path+'dynamics')
        if self.encoder:
            torch.save(self.encoder.state_dict(), path+'encoder')
        if self.decoder:    
            torch.save(self.decoder.state_dict(), path+'decoder')

    def load_model(self, path):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.dynamics_model:
            self.dynamics_model.load_state_dict(torch.load(path+'dynamics', map_location=device))
            self.dynamics_model.eval()
            print('Load dynamics model.')
        if self.encoder:    
            self.encoder.load_state_dict(torch.load(path+'encoder', map_location=device))
            self.encoder.eval()
            print('Load encoder.')
        if self.decoder:
            self.decoder.load_state_dict(torch.load(path+'decoder', map_location=device))
            self.decoder.eval()
            print('Load decoder.')

class EncoderDynamicsNetwork(EmbeddingDynamicsNetwork):
    """ Dyanmics prediction network with dynamics embedding using encoder only, no decoder regularization """
    def __init__(self, state_space, action_space, param_dim, latent_dim,
    hidden_dim=256, hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr1=1e-3, lr2=1e-4, gamma=0.99):
        super().__init__()
        self.dynamics_model = DynamicsNetwork(state_space, action_space, latent_dim, hidden_dim, hidden_activation, output_activation, num_hidden_layers, lr=lr1, gamma=gamma)
        self.encoder = DynamicsEncoder(param_dim, latent_dim)
        self.optimizer1 = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr1)
        self.optimizer2 = torch.optim.Adam(self.encoder.parameters(), lr=lr2)

    def forward(self, sa, theta):
        if not isinstance(sa, torch.Tensor):
            sa = torch.Tensor(sa)
        if not isinstance(theta, torch.Tensor):
            theta = torch.Tensor(theta)        

        # encoder
        self.alpha = self.encoder(theta)

        # dynamics prediction
        x=torch.cat((sa, self.alpha), axis=-1)
        pre_s_ = self.dynamics_model(x)

        return None, pre_s_

class EncoderDecoderDynamicsNetwork(EmbeddingDynamicsNetwork):
    """ Dyanmics prediction network with dynamics embedding using encoder and decoder (as regularization) """
    def __init__(self, state_space, action_space, param_dim, latent_dim,
    hidden_dim=256, hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr1=1e-3, lr2=5e-3, gamma=0.99):
        super().__init__()
        self.dynamics_model = DynamicsNetwork(state_space, action_space, latent_dim, hidden_dim, hidden_activation, output_activation, num_hidden_layers, lr=lr1, gamma=gamma)
        self.encoder = DynamicsEncoder(param_dim, latent_dim)
        self.decoder = DynamicsDecoder(param_dim, latent_dim)
        self.optimizer1 = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr1)
        self.optimizer2 = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=lr2)

    def forward(self, sa, theta):
        if not isinstance(sa, torch.Tensor):
            sa = torch.Tensor(sa)
        if not isinstance(theta, torch.Tensor):
            theta = torch.Tensor(theta)        

        # encoder-decoder
        self.alpha = self.encoder(theta)
        pre_theta_ = self.decoder(self.alpha)

        # dynamics prediction
        x=torch.cat((sa, self.alpha), axis=-1)
        pre_s_ = self.dynamics_model(x)

        return pre_theta_, pre_s_

class VAEDynamicsNetwork(EmbeddingDynamicsNetwork):
    """ Dyanmics prediction network with dynamics embedding using variational auto-encoder and decoder (as regularization) """
    def __init__(self, state_space, action_space, param_dim, latent_dim,
    hidden_dim=256, hidden_activation=F.relu, output_activation=None, num_hidden_layers=6, lr1=1e-3, lr2=5e-4, gamma=0.99):
        super().__init__()
        self.dynamics_model = DynamicsNetwork(state_space, action_space, latent_dim, hidden_dim, hidden_activation, output_activation, num_hidden_layers, lr=lr1, gamma=gamma)
        self.encoder = DynamicsVariationalEncoder(param_dim, latent_dim)
        self.decoder = DynamicsDecoder(param_dim, latent_dim)
        self.optimizer1 = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr1)
        self.optimizer2 = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=lr2)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer2, gamma=0.99)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps*std+mu
        return z
    
    def loss_vae(self, recon_x, x):
        # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        BCE = self.loss_recon(recon_x, x)
        KLD = torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(), axis=-1))
        # print(BCE, KLD)
        return BCE + 0.01*KLD, BCE, KLD  # 0.1 for halfcheetah and 0.01 for pandapushfk!

    def forward(self, sa, x):
        if not isinstance(sa, torch.Tensor):
            sa = torch.Tensor(sa)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)  

        # encoder-decoder
        self.mu, self.logvar = self.encoder(x)
        z = self.reparametrize(self.mu, self.logvar)
        x_hat = self.decoder(z)

        # dynamics prediction
        d=torch.cat((sa, self.mu), axis=-1)
        pre_s_ = self.dynamics_model(d)
        return x_hat, pre_s_
