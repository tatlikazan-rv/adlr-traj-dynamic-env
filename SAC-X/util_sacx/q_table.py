import collections

from .q_estimator import QEstimator


class QTable(collections.MutableMapping, QEstimator):
    """
        Q Table implementation
    """

    def __init__(self, *args, **kwargs):
        """
        Create a new Q Table
        """
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key: tuple) -> float:
        """
        Get the Q-value corresponding to the given key
        :param key: A two-tuple of (state, action)
        :return: The Q-value corresponding to the (state, action) pair. Return 0 if the pair is not in this table
        """
        s, a = key
        return self.store.get(s, dict()).get(a, 0)

    def __setitem__(self, key: tuple, value: float):
        """
        Set a Q-value for a given (state, action) pair
        :param key: Two-tuple of (state, action)
        :param value: Q-value corresponding to the key
        """
        s, a = key
        self.store.setdefault(s, dict())[a] = value

    def __delitem__(self, key: tuple):
        """
        Remove an entry from this table
        :param key: The (state, action) pair of the entry that should be removed
        """
        s, a = key
        del self.store[s][a]

    def __iter__(self):
        """
        :return: An iterator that iterates through all (state, action) pairs stored in this table
        """
        for state, actions in self.store.items():
            for action in iter(actions):
                yield (state, action)

    def __len__(self) -> int:
        """
        :return: The number of entries in this table
        """
        return sum([len(actions) for actions in self.store.values()])

    def __str__(self) -> str:
        """
        :return: A pretty string representation  TODO -- fix
        """
        t = '| Q Table                                    Value    |\n'
        t += '+------------------------------------------+----------+\n'
        t_entry = '| {:<40} | {:8.3f} |\n'
        for s, a in self:
            t += t_entry.format(str(s) + ', ' + str(a), self[s, a])
        t += '+------------------------------------------+----------+\n'
        return t

    def Q(self, state, action):
        """
        Obtain the Q-value for performing the specified action given the state
        :param state: The obtained state
        :param action: The action to be performed
        :return: the corresponding Q-value
        """
        return self[state, action]

    def Qs(self, state, actions):
        """
        Obtain all Q-values for multiple possible actions given the state
        :param state: The obtained state
        :param actions: A list of actions for which the Q-value should be obtained
        :return: A dictionary mapping each action to the corresponding Q-value
        """
        return {a: self[state, a] for a in actions}

