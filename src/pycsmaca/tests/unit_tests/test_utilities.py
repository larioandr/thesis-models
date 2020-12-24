import pytest

from pycsmaca.utilities import ReadOnlyDict


##############################################################################
# TEST ReadOnlyDict
##############################################################################
def test_read_only_dict_get_values_but_does_not_allow_change_them():
    data = {'one': 'hello', 2: 'something'}
    rod = ReadOnlyDict(data)

    assert 'one' in rod
    assert 2 in rod
    assert 0 not in rod

    assert rod['one'] == 'hello'
    assert rod[2] == 'something'

    with pytest.raises(TypeError):
        rod[0] = 'new key can not be added'
    with pytest.raises(TypeError):
        rod[2] = 'value can not be changed'


def test_read_only_dict_proxy_getters_from_dict():
    data = {'one': 'hello', 2: 'something'}
    rod = ReadOnlyDict(data)

    assert rod.items() == data.items()
    assert tuple(rod.values()) == tuple(data.values())
    assert rod.keys() == data.keys()


def test_read_only_dict_can_be_compared():
    data = {'one': 'hello', 2: 'something'}
    first_rod = ReadOnlyDict(data)
    second_rod = ReadOnlyDict(data.copy())

    assert first_rod == second_rod
    assert first_rod == data
    assert first_rod != {0: 'wrong!'}
    assert not (first_rod != data)


def test_read_only_dict_copy_is_also_read_only():
    data = {'one': 'hello', 2: 'something'}
    rod = ReadOnlyDict(data)

    assert isinstance(rod.copy(), ReadOnlyDict)
    assert rod.copy() == rod


def test_read_only_dict_is_iterable():
    data = {'one': 'hello', 2: 'something'}
    rod = ReadOnlyDict(data)

    keys = set()
    for key in rod:
        keys.add(key)

    assert set(rod.keys()) == keys


def test_read_only_dict_provides_str():
    data = {'one': 'hello', 2: 'something'}
    rod = ReadOnlyDict(data)

    assert str(rod) == ('RODict' + str(data))
