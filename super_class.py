
class Test:
    # structure
    def __new__(cls, *args, **kwargs):
        """
        Reason:  to create a new object
        When called:  first to create new object
        :param args: normal arguments
        :param kwargs: keyworded arguments (a=1)
        """
        pass

    def __init__(self):
        """
        Reason:  initialize variables and do other actions once created
        When called:  immediedtaly after __new__ when there is a new object
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Reason:  perform a action with certain arguments - 2nd __init__
        When called:  object with parenthesis - object(1, 2, 3)
        :param args: args
        :param kwargs: kwargs
        """
        pass

    def __init_subclass__(cls, **kwargs):
        """
        called when a class is subclassed to initialize its variables
        :param kwargs: idk
        """
        pass

    def __hash__(self):
        """
        do stuff to create a unique code
        :return: a number
        """
        pass

    def __dir__(self):
        """
        The directory is a dictionary of all attributes and their values
        :return: the dictionary
        """
        pass

    def __class__(self):
        pass

    def __repr__(self):
        pass

    # deletion

    def __delete__(self, instance):
        pass

    def __del__(self):
        pass

    def __delattr__(self, item):
        pass

    def __delslice__(self, i, j):
        pass

    # IO
    def __get__(self, instance, owner):
        pass

    def __getattr__(self, item):
        pass

    def __getattribute__(self, item):
        pass

    def __getinitargs__(self):
        pass

    def __getnewargs__(self):
        pass

    def __getstate__(self):
        pass

    def __set__(self, instance, value):
        pass

    def __setattr__(self, key, value):
        pass

    def __set_name__(self, owner, name):
        pass

    def __setslice__(self, i, j, sequence):
        pass

    def __setstate__(self, state):
        pass

    # comparators
    def __eq__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):  # >=
        pass

    def __le__(self, other):
        pass

    def __ne__(self, other):
        pass

    def __cmp__(self, other):  # false = neg, eq = 0, true = pos
        pass

    def __and__(self, other):
        pass

    def __or__(self, other):
        pass

    def __xor__(self, other):
        pass

    # alter equals
    def __iadd__(self, other):
        pass

    def __iand__(self, other):
        pass

    def __idiv__(self, other):
        pass

    def __ifloordiv__(self, other):
        pass

    def __ilshift__(self, other):
        pass

    def __irshift__(self, other):
        pass

    def __imatmul__(self, other):
        pass

    def __imod__(self, other):
        pass

    def __imul__(self, other):
        pass

    def __ior__(self, other):
        pass

    def __ixor__(self, other):
        pass

    def __ipow__(self, other):
        pass

    def __isub__(self, other):
        pass

    def __itruediv__(self, other):
        pass

    # math
    def __pos__(self):
        pass

    def __neg__(self):
        pass

    def __invert__(self):
        pass

    def __round__(self, n=None):
        pass

    def __floor__(self):
        pass

    def __ceil__(self):
        pass

    def __trunc__(self):
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __abs__(self):
        pass

    def __pow__(self, power, modulo=None):
        pass

    def __divmod__(self, other):
        pass

    def __floordiv__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __lshift__(self, other):
        pass

    def __rshift__(self, other):
        pass

    def __matmul__(self, other):
        pass

    def __mod__(self, other):
        pass

    # reflected operations
    def __radd__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __rfloordiv__(self, other):
        pass

    def __rdiv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __rmod__(self, other):
        pass

    def __rdivmod__(self, other):
        pass

    def __rlshift__(self, other):
        pass

    def __rrshift__(self, other):
        pass

    def __rand__(self, other):
        pass

    def __ror__(self, other):
        pass

    def __rxor__(self, other):
        pass

    def __rmatmul__(self, other):
        pass

    def __rpow__(self, other):
        pass

    # typing
    def __str__(self):
        pass

    def __bool__(self):
        pass

    def __bytes__(self):
        pass

    def __int__(self):
        pass

    def __float__(self):
        pass

    def __complex__(self):
        pass

    def __oct__(self):
        pass

    def __hex__(self):
        pass

    def __coerce__(self, other):
        pass

    def __long__(self):
        pass

    # asynchronous
    def __aenter__(self):
        pass

    def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __aiter__(self):
        pass

    def __anext__(self):
        pass

    def __await__(self):
        pass

    # context managers - "with" command
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # copies
    def __copy__(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass

    # information


    # sequences
    def __len__(self):
        pass

    def __index__(self):
        pass

    def __iter__(self):
        pass

    def __contains__(self, item):
        pass

    def __missing__(self, key):
        pass

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    # def __class__(self: _T) -> Type[_T]:
    #     pass

    def __unicode__(self):
        pass

    def __format__(self, format_spec):
        pass

    def __sizeof__(self):
        pass

    def __instancecheck__(self, instance):
        pass

    def __subclasscheck__(self, subclass):
        pass

    def __reduce__(self):
        pass

    def __reduce_ex__(self, protocol):
        pass

    def __fspath__(self):
        pass

    def __next__(self):
        pass

    @classmethod
    def __prepare__(metacls, name, bases):
        pass

    def __reversed__(self):
        pass


if __name__ == '__main__':
    t = Test()
    print(t)