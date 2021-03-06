import phe as paillier
from .paillier import EncryptedNumber
import numpy as np
import pickle
import syft
from .basic import Float, PaillierTensor
from ...tensor import TensorBase
from ..abstract.keys import AbstractSecretKey, AbstractPublicKey
from ..abstract.keys import AbstractKeyPair


class SecretKey(AbstractSecretKey):

    def __init__(self, sk):
        self.sk = sk

    def decrypt(self, x):
        """Decrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""

        if(type(x) == Float):
            return self.sk.decrypt(x.data)
        elif(type(x) == TensorBase or type(x) == PaillierTensor):
            if(x.encrypted):
                return TensorBase(self.decrypt(x.data), encrypted=False)
            else:
                return NotImplemented
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                if (type(v) == EncryptedNumber):
                    out.append(self.sk.decrypt(v))
                    print('vvvvvv',type(v))
                else:
                    out.append(self.sk.decrypt(v.data))
            return np.array(out).reshape(sh)
        else:
            return NotImplemented

    def transform(self,x):
        if(type(x) == Float):
            return self.sk.transform(x.data)
        elif(type(x) == EncryptedNumber):
            return PaillierTensor(self.sk.transform(x))
        elif(type(x) == TensorBase or type(x) == PaillierTensor):
            return PaillierTensor(x.public_key,self.transform(x.data),False)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(self.transform(v))
            return np.array(out).reshape(sh)
        else:
            return NotImplemented

    def finalDecrypt(self, x):
        """Decrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""

        if(type(x) == Float):
            return self.sk.finalDecrypt(x.data)
        elif(type(x) == EncryptedNumber):
            return self.sk.finalDecrypt(x)
        elif(type(x) == TensorBase or type(x) == PaillierTensor):
            return PaillierTensor(x.public_key,self.finalDecrypt(x),False)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(self.finalDecrypt(v))
            return np.array(out).reshape(sh)
        else:
            return NotImplemented

    def serialize(self):
        return pickle.dumps(self.sk)

    def deserialize(b):
        return SecretKey(pickle.loads(b))


class PublicKey(AbstractPublicKey):

    def __init__(self, pk):
        self.pk = pk

    def zeros(self, dim):
        """Returns an encrypted tensor of zeros"""
        return syft.zeros(dim).encrypt(self)

    def ones(self, dim):
        """Returns an encrypted tensor of ones"""
        return syft.ones(dim).encrypt(self)

    def rand(self, dim):
        """Returns an encrypted tensor with initial numbers sampled from a
        uniform distribution from 0 to 1."""
        return syft.rand(dim).encrypt(self)

    def randn(self, dim):
        """Returns an encrypted tensor with initial numbers sampled from a
        standard normal distribution"""
        return syft.randn(dim).encrypt(self)

    def encrypt(self, x, same_type=False):
        """Encrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""
        if(type(x) == int or type(x) == float or type(x) == np.float64):
            if(same_type):
                return NotImplemented
            return Float(self, x)
        elif(type(x) == TensorBase):
            if(x.encrypted or same_type):
                return NotImplemented
            return PaillierTensor(self, x.data)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(Float(self, v))
            if(same_type):
                return np.array(out).reshape(sh)
            else:
                return PaillierTensor(self, np.array(out).reshape(sh))
        else:
            print("format not recognized:" + str(type(x)))
            return NotImplemented

        return self.pk.encrypt(x)

    def serialize(self):
        return pickle.dumps(self.pk)

    def deserialize(b):
        return PublicKey(pickle.loads(b))


class KeyPair(AbstractKeyPair):

    def __init__(self):
        ""

    def deserialize(self, pubkey, seckey):
        self.public_key = PublicKey(pickle.loads(pubkey))
        self.secret_key = SecretKey(pickle.loads(seckey))
        return (self.public_key, self.secret_key)

    def generate(self, n_length=2048):
        pubkey, prikey = paillier.generate_paillier_keypair(n_length=n_length)
        self.public_key = PublicKey(pubkey)
        self.secret_key = SecretKey(prikey)

        return (self.public_key, self.secret_key)
