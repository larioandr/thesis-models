from enum import Enum
import types
from .base import Singleton
from . import errors


class FType(Enum):
    """
    Function type. Could be initializer, finalizer, event handler or
    any other (native, not managed) function.
    """
    NOT_MANAGED = 0
    INITIALIZER = 1
    EVENT_HANDLER = 2
    FINALIZER = 3
    STOP_CONDITION = 4
    DEATH_CONDITION = 5
    MANAGED_METHOD = 6


def get_sim_name(obj):
    """
    Get an in-simulation object name: if ``obj`` has attribute
    ``__sim_name__``, it will be returned, otherwise ``__name__``
    standard attribute.

    Args:
        obj: an object to get name of

    Returns: object name
    """
    if hasattr(obj, '__sim_name__') and obj.__sim_name__ is not None:
        return obj.__sim_name__
    else:
        return obj.__name__


def define_sim_function(fn, ftype, guard=None, default_handler=None,
                        sim_name=None, stage=None):
    """
    Adds attributes to the function or method to make it manageable by
    the simulation system.

    Args:
        fn: a function to manage

        ftype: function type, see ``FType``

        guard: optional condition - a function with a signature,
            depending on the ``ftype``.

        default_handler: optional boolean flag, indicating the function
            will be the default event handler, applicable when for
            ``ftype = FType.EVENT_HANDLER``

        sim_name: user-readable function name. If not set, the default
            ``fn.__name__`` will be used as the function name.

        stage: optional stage (for initializers and finalizers) - a
            function priority, defining the order in which initializers
            and finalizers are called.

    Returns:

    """
    if ftype is FType.NOT_MANAGED:
        raise errors.IllegalValueError('ftype', ftype.name)
    fn.__sim_ftype__ = ftype
    fn.__sim_name__ = sim_name
    fn.get_name = types.MethodType(get_sim_name, fn)
    if ftype in [FType.STOP_CONDITION, FType.DEATH_CONDITION]:
        fn.__sim_guard__ = guard
    elif ftype in [FType.INITIALIZER, FType.FINALIZER]:
        fn.__sim_stage__ = stage
    elif ftype is FType.EVENT_HANDLER:
        fn.__sim_guard__ = guard
        fn.__sim_default_handler__ = bool(default_handler)


def sort(items, key=None, start_with_none=False, reverse=False):
    """
    Sorts a list (or any other iterable). It differs from built-in
    function ``sorted`` by treating ``None`` keys in a special way:
    they are all gathered either in the end of the result (default),
    or in the beginning. It also constructs the list and returns it
    completely instead of yielding.

    Args:
        items: any finite iterable

        key: optional, function ``item -> bool``

        start_with_none: if ``True``, items with ``None`` keys will
            go first.

        reverse: sort in the decrementing order

    Returns: a sorted list
    """
    if key is not None:
        not_nones = [item for item in items if key(item) is not None]
        nones = [item for item in items if key(item) is None]
    else:
        not_nones = [item for item in items if item is not None]
        nones = [item for item in items if item is None]
    not_nones = sorted(not_nones, key=key, reverse=reverse)
    return nones + not_nones if start_with_none else not_nones + nones


def find_fired_conditions(conditions, guard=None, *args, **kwargs):
    """
    For an iterable (e.g. list) of boolean functions, find a list of
    functions returning ``True``.

    If ``guard`` is given, it is applied to a function to get the
    predicate - a function ``() -> bool``. If this predicate is
    not ``None``, it is checked and the condition is then evaluated
    if and only if the predicate returns ``True``. If ``guard`` is
    not provided, or the predicate is ``None``, condition is tested
    without additional check. Normally the predicate should be a
    very short function allowing to test whether a complex condition
    need to be evaluated.

    Args:
        conditions: an iterable of boolean functions

        guard: a ``(condition) -> predicate`` function, where
            ``predicate`` is ``() -> bool``.

        *args: positional arguments passed to each condition

        **kwargs: keyword arguments passed to each condition

    Returns: a list of conditions evaluated to ``True``
    """
    fired = []
    if guard is not None:
        for condition in conditions:
            g = guard(condition)
            if g is not None and g():
                if condition(*args, **kwargs):
                    fired.append(condition)
            else:
                if condition(*args, **kwargs):
                    fired.append(condition)
    return fired


class CallStack(object, metaclass=Singleton):
    """
    Object groups data about current running managed functions, e.g.
    handlers, initializers, etc.
    """

    def __init__(self):
        self.call_stack = []

    def push(self, func, entity=None, ft=FType.EVENT_HANDLER):
        self.call_stack.append((entity, func, ft))

    def pop(self):
        return self.call_stack.pop()

    def __bool__(self):
        return bool(self.call_stack)

    @property
    def function(self):
        return self.call_stack[-1][1] if self.call_stack else None

    @property
    def entity(self):
        return self.call_stack[-1][0] if self.call_stack else None

    @property
    def ftype(self):
        return self.call_stack[-1][2] if self.call_stack else None

    def clear(self):
        self.call_stack = []


#######################################################################
# ENTITIES
#
# Entities are the modules defined by the users. They can:
# 1) initialize
# 2) handle events
# 3) finalize
# 4) define stop conditions
# 5) define death conditions
#
# Defines:
# + MetaEntity metaclass
# + Entity class
# + decorators for initializing, event-handling and finalizing methods
#######################################################################

class MetaEntity(type):
    """
    A metaclass for all objects those could be used as simulation
    entities. When a class with this metaclass is defined, this class
    constructor (defined in this metaclass) processes all fields and
    methods to find initializers, finalizers, event handlers, stop
    and death conditions.

    A standard way to define a new entity is to subclass from
    ``Entity`` class (define in this module) which already has
    ``Managed`` metaclass set.
    """
    __sim_initializers__ = []
    __sim_finalizers__ = []
    __sim_handlers__ = []
    __sim_stop_conditions__ = []
    __sim_death_conditions__ = []
    __sim_default_handler__ = None

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        # The following code is needed to make sure derived
        # classes' objects would have their own methods lists
        # with inherited methods from parents
        # (otherwise children and parents would share the same
        # lists, so adding something to any child would cause
        # parent's methods list modification)

        cls.__sim_initializers__ = [] + cls.__sim_initializers__
        cls.__sim_finalizers__ = [] + cls.__sim_finalizers__
        cls.__sim_handlers__ = [] + cls.__sim_handlers__
        cls.__sim_stop_conditions__ = [] + cls.__sim_stop_conditions__
        cls.__sim_death_conditions__ = [] + cls.__sim_death_conditions__

        inherited_default_handler = cls.__sim_default_handler__
        cls.__sim_default_handler__ = None

        for fname, f in attrs.items():
            if not hasattr(f, '__sim_ftype__') or \
                            f.__sim_ftype__ is FType.NOT_MANAGED:
                continue
            ft = f.__sim_ftype__
            if ft is FType.INITIALIZER:
                if f not in cls.__sim_initializers__:
                    cls.__sim_initializers__.append(f)
            elif ft is FType.FINALIZER:
                if f not in cls.__sim_finalizers__:
                    cls.__sim_finalizers__.append(f)
            elif ft is FType.EVENT_HANDLER:
                if f not in cls.__sim_handlers__:
                    cls.__sim_handlers__.append(f)
                    if (hasattr(f, '__sim_default_handler__') and
                            f.__sim_default_handler__):
                        if cls.__sim_default_handler__ is not None:
                            raise errors.AmbiguousDefaultHandler(cls)
                        cls.__sim_default_handler__ = f
            elif ft is FType.STOP_CONDITION:
                if f not in cls.__sim_stop_conditions__:
                    cls.__sim_stop_conditions__.append(f)
            elif ft is FType.DEATH_CONDITION:
                if f not in cls.__sim_death_conditions__:
                    cls.__sim_death_conditions__.append(f)

        if cls.__sim_default_handler__ is None:
            cls.__sim_default_handler__ = inherited_default_handler

        # print("********** MetaEntity <- {}".format(cls))
        # print("* handlers        : {}".format(cls.__sim_handlers__))
        # print("* stop conditions : {}".format(cls.__sim_stop_conditions__))
        # print("* death conditions: {}".format(cls.__sim_death_conditions__))


class Entity(object, metaclass=MetaEntity):
    def __init__(self):
        super().__init__()

    @staticmethod
    def eventhandler(guard=None, default=False, name=None):
        """
        Produce an event handler from the method.

        Args:
            guard: a guard for the event. Any function with signature
                ``event, source -> bool``, where ``source`` is the entity
                that created the event.

            default: if set to ``True``, any event that was sent to this
                entity but doesn't match any predicate would be processed
                by this handler. Only one handler may be marked as default.

            name: human-readable method name. If left None, method name
                will be used.

        Returns: a decorator
        """
        def decorator(method):
            def wrapper(self, event, source):
                CallStack().push(wrapper, self, FType.EVENT_HANDLER)
                ret = method(self, event, source)
                CallStack().pop()
                return ret

            # noinspection PyTypeChecker
            define_sim_function(
                wrapper, FType.EVENT_HANDLER, guard=guard,
                default_handler=default,
                sim_name=name if name is not None else method.__name__)
            return wrapper
        return decorator

    @staticmethod
    def initializer(stage=0, name=None):
        """
        Produces an initializer.

        Args:
            stage: number of stage, can be of any comparable type

            name: user-defined function name, optional

        Returns: a decorator
        """
        def decorator(method):
            def wrapper(self):
                CallStack().push(wrapper, self, FType.INITIALIZER)
                ret = method(self)
                CallStack().pop()
                return ret

            # noinspection PyTypeChecker
            define_sim_function(
                wrapper, FType.INITIALIZER, stage=stage,
                sim_name=name if name is not None else method.__name__)
            return wrapper
        return decorator

    @staticmethod
    def finalizer(stage=0, name=None):
        """
        Produces a finalizer.

        Args:
            stage: number of stage, can be of any comparable type

            name: user-defined function name, optional

        Returns: a decorator
        """
        def decorator(method):
            def wrapper(self):
                CallStack().push(wrapper, self, FType.FINALIZER)
                ret = method(self)
                CallStack().pop()
                return ret

            # noinspection PyTypeChecker
            define_sim_function(
                wrapper, FType.FINALIZER, stage=stage,
                sim_name=name if name is not None else method.__name__)
            return wrapper
        return decorator

    @staticmethod
    def stop_condition(guard=None, name=None):
        """
        Produces a stop condition

        Args:
            guard: a simple lambda ``() -> bool``

            name: user-defined function name, optional

        Returns: a decorator
        """
        def decorator(method):
            def wrapper(self):
                CallStack().push(wrapper, self, FType.STOP_CONDITION)
                ret = method(self)
                CallStack().pop()
                return ret

            # noinspection PyTypeChecker
            define_sim_function(
                wrapper, FType.STOP_CONDITION, guard=guard,
                sim_name=name if name is not None else method.__name__)
            return wrapper
        return decorator

    @staticmethod
    def death_condition(guard=None, name=None):
        """
        Produces a death condition

        Args:
            guard: a simple lambda ``() -> bool``. If set and not satisfied,
                actual condition is not checked

            name: user-defined function name, optional

        Returns: a decorator
        """
        def decorator(method):
            def wrapper(self):
                CallStack().push(wrapper, self, FType.DEATH_CONDITION)
                ret = method(self)
                CallStack().pop()
                return ret

            # noinspection PyTypeChecker
            define_sim_function(
                wrapper, FType.DEATH_CONDITION, guard=guard,
                sim_name=name if name is not None else method.__name__)
            return wrapper
        return decorator

    @staticmethod
    def managed(method):
        def wrapper(self, *args, **kwargs):
            CallStack().push(wrapper, self, FType.MANAGED_METHOD)
            ret = method(self, *args, **kwargs)
            CallStack().pop()
            return ret
        # noinspection PyTypeChecker
        define_sim_function(wrapper, FType.MANAGED_METHOD,
                            sim_name=method.__name__)
        return wrapper

    @classmethod
    def get_sorted_initializers(cls):
        return sort(cls.__sim_initializers__, key=lambda f: f.__sim_stage__)

    @classmethod
    def get_sorted_finalizers(cls):
        return sort(cls.__sim_finalizers__, key=lambda f: f.__sim_stage__)

    @classmethod
    def get_event_handler(cls, event, source):
        for f in cls.__sim_handlers__:
            if hasattr(f, '__sim_guard__') and f.__sim_guard__ is not None:
                if f.__sim_guard__(event, source):
                    return f
        if bool(cls.__sim_default_handler__):
            return cls.__sim_default_handler__
        raise errors.EventHandlerNotFound(event, source, cls.__name__)

    def check_stop_conditions(self):
        conditions = self.__class__.__sim_stop_conditions__
        return find_fired_conditions(conditions, lambda f: f.__sim_guard__,
                                     self)

    def check_death_conditions(self):
        conditions = self.__class__.__sim_death_conditions__
        return find_fired_conditions(conditions, lambda f: f.__sim_guard__,
                                     self)


class EntitiesRegistry(object):
    class Stage(Enum):
        NEW = 0
        INITIALIZED = 1
        FINISHED = 2

    def __init__(self):
        self._entities = {}

    def clear(self):
        self._entities = {}

    def entities(self, stage=None):
        """
        List registered entities. If ``stage`` provided, list only
        entities in the specific stage. If not, new entities will
        be listed before initialized entities, and finished entities
        will go in the end.

        Args:
            stage:

        Returns: a list of entities
        """
        if stage is None:
            return self.entities(self.Stage.NEW) + \
                   self.entities(self.Stage.INITIALIZED) + \
                   self.entities(self.Stage.FINISHED)
        else:
            return [k for k, v in self._entities.items() if v == stage]

    def add(self, entity):
        """
        Adds an entity in the stage ``NEW``.

        Args:
            entity: an entity to add

        Raises:
            ``pyons.errors.EntityAlreadyExists`` if entity is already
            registered.
        """
        if entity in self._entities:
            raise errors.EntityAlreadyExists(entity)
        self._entities[entity] = self.Stage.NEW

    def remove(self, entity):
        """
        Remove an entity. If it doesn't exist, do nothing.

        Args:
            entity: what to delete.
        """
        if entity in self._entities:
            del self._entities[entity]

    def contains(self, entity, stage=None):
        """
        Test whether an entity is registered and, optionally, is under
        a specified stage.

        Args:
            entity: an object to test for

            stage: if provided, also checks that the entity is under
                the given stage.

        Returns: ``bool``
        """
        if stage is None:
            return entity in self._entities
        else:
            return entity in self._entities and self._entities[entity] == stage

    def stage(self, entity):
        """
        Get the entity stage.

        Args:
            entity: an object to test for

        Raises:
            ``pyons.errors.EntityNotFoundError``

        Returns: ``EntitiesRegistry.Stage``
        """
        if entity in self._entities:
            return self._entities[entity]
        else:
            raise errors.EntityNotFoundError(entity)

    def set_stage(self, entity, stage):
        """
        Set the entity stage.

        Args:
            entity: an entity to set stage for.

            stage: a stage to set.

        Raises:
            ``pyons.errors.EntityNotFoundError``
        """
        if entity in self._entities:
            self._entities[entity] = stage
        # else:
        #     print(">>>>>" + entity)
        #     raise errors.EntityNotFoundError(entity)


#######################################################################
# GLOBAL (STATIC) ENTITIES, INITIALIZERS, FINALIZERS AND HANDLERS
#
# The global (or static) handlers are defined as regular (non-method)
# functions. They are registered within StaticRegistry singleton.
#######################################################################

class StaticRegistry(object, metaclass=Singleton):
    def __init__(self):
        super().__init__()
        self.__initializers__ = []
        self.__finalizers__ = []
        self.__stop_conditions__ = []
        self.__event_handlers__ = []
        self.__default_event_handler__ = None

    def register(self, function):
        if not hasattr(function, '__sim_ftype__') \
                or function.__sim_ftype__ is FType.NOT_MANAGED:
            raise errors.IllegalValueError(
                'function', function,
                "function doesn't have __sim_ftype__ parameter "
                "or it is NOT_MANAGED")
        ft = function.__sim_ftype__
        if ft is FType.DEATH_CONDITION:
            raise errors.IllegalValueError(
                'function.__sim_ftype__', ft.name,
                "static death conditions disallowed")
        elif ft is FType.INITIALIZER:
            self.__initializers__.append(function)
        elif ft is FType.FINALIZER:
            self.__finalizers__.append(function)
        elif ft is FType.EVENT_HANDLER:
            self.__event_handlers__.append(function)
            if hasattr(function, '__sim_default_handler__') \
                    and bool(function.__sim_default_handler__):
                if self.__default_event_handler__ is not None:
                    raise errors.AmbiguousDefaultHandler()
                self.__default_event_handler__ = function

        elif ft is FType.STOP_CONDITION:
            self.__stop_conditions__.append(function)

    def get_sorted_initializers(self):
        return sort(self.__initializers__, key=lambda f: f.__sim_stage__)

    def get_sorted_finalizers(self):
        return sort(self.__finalizers__, key=lambda f: f.__sim_stage__)

    def get_event_handler(self, event, source):
        for f in self.__event_handlers__:
            if hasattr(f, '__sim_guard__') and f.__sim_guard__ is not None:
                if f.__sim_guard__(event, source):
                    return f
        if bool(self.__default_event_handler__):
            return self.__default_event_handler__
        raise errors.EventHandlerNotFound(event, source, None)

    def check_stop_conditions(self):
        return find_fired_conditions(
            self.__stop_conditions__, lambda f: f.__sim_guard__)


def eventhandler(guard=None, default=False, name=None):
    """
    Produce an event handler from the static function.

    Args:
        guard: a guard for the event. Any function with signature
            ``event, source -> bool``, where ``source`` is the entity
            that created the event.

        default: if set to ``True``, any event that was sent without
            target entity specified and doesn't match any predicate
            would be processed by this handler. Only one handler may
            be marked as default (while each entity may have its
            own default handler)

        name: human-readable name. If left None, function native name
            will be used.

    Returns: a decorator
    """

    # noinspection PyTypeChecker
    def decorator(fn):
        def wrapper(event, source):
            CallStack().push(wrapper, None, FType.EVENT_HANDLER)
            ret = fn(event, source)
            CallStack().pop()
            return ret

        define_sim_function(wrapper, FType.EVENT_HANDLER, guard=guard,
                            sim_name=name if name is not None else fn.__name__)
        StaticRegistry().register(wrapper)
        return wrapper
    return decorator


def initializer(stage=0, name=None):
    """
    Produces an initializer.

    Args:
        stage: number of stage, can be of any comparable type

        name: user-defined function name, optional

    Returns: a decorator
    """

    # noinspection PyTypeChecker
    def decorator(fn):
        def wrapper():
            CallStack().push(wrapper, None, FType.INITIALIZER)
            ret = fn()
            CallStack().pop()
            return ret

        define_sim_function(wrapper, FType.INITIALIZER, stage=stage,
                            sim_name=name if name is not None else fn.__name__)
        StaticRegistry().register(wrapper)
        return wrapper
    return decorator


def finalizer(stage=0, name=None):
    """
    Produces a finalizer.

    Args:
        stage: number of stage, can be of any comparable type

        name: user-defined function name, optional

    Returns: a decorator
    """

    # noinspection PyTypeChecker
    def decorator(fn):
        def wrapper():
            CallStack().push(wrapper, None, FType.FINALIZER)
            ret = fn()
            CallStack().pop()
            return ret

        define_sim_function(wrapper, FType.FINALIZER, stage=stage,
                            sim_name=name if name is not None else fn.__name__)
        StaticRegistry().register(wrapper)
        return wrapper
    return decorator


def stop_condition(guard=None, name=None):
    """
    Produces a stop condition

    Args:
        guard: a simple lambda ``() -> bool``. If set, this guard will
            be checked before running the decorated function. If
            evaluated to ``False``, the decorated function won't be run

        name: user-defined function name, optional

    Returns: a decorator
    """

    # noinspection PyTypeChecker
    def decorator(fn):
        def wrapper():
            CallStack().push(wrapper, None, FType.STOP_CONDITION)
            ret = fn()
            CallStack().pop()
            return ret

        define_sim_function(wrapper, FType.STOP_CONDITION, guard=guard,
                            sim_name=name if name is not None else fn.__name__)
        StaticRegistry().register(wrapper)
        return wrapper
    return decorator


######################################################################
# HELPERS FOR EVENT HANDLING, INITIALIZATION, FINALIZATION, etc.
######################################################################
def initialize(entities_registry=None, static_registry=None, entity=None):
    if entity is not None:
        assert isinstance(entities_registry, EntitiesRegistry)
        if entities_registry.stage(entity) is EntitiesRegistry.Stage.NEW:
            for f in entity.get_sorted_initializers():
                f(entity)
            entities_registry.set_stage(
                entity, EntitiesRegistry.Stage.INITIALIZED)
    else:
        assert isinstance(entities_registry, EntitiesRegistry)
        assert isinstance(static_registry, StaticRegistry)
        entities = entities_registry.entities(EntitiesRegistry.Stage.NEW)
        fs = [(f.__sim_stage__, f, entity) for entity in entities
              for f in entity.__sim_initializers__]
        fs += [(f.__sim_stage__, f, None)
               for f in static_registry.__initializers__]
        fs = sort(fs, key=lambda x: x[0])
        for stage, f, entity in fs:
            if entity is None:
                f()
            else:
                f(entity)
        for entity in entities:
            entities_registry.set_stage(
                entity, EntitiesRegistry.Stage.INITIALIZED)


def finalize(entities_registry=None, static_registry=None, entity=None):
    if entity is not None:
        assert isinstance(entities_registry, EntitiesRegistry)
        stage = entities_registry.stage(entity)
        if stage is EntitiesRegistry.Stage.INITIALIZED:
            for f in entity.get_sorted_finalizers():
                f(entity)
            entities_registry.set_stage(
                entity, EntitiesRegistry.Stage.FINISHED)
    else:
        assert isinstance(entities_registry, EntitiesRegistry)
        assert isinstance(static_registry, StaticRegistry)
        entities = entities_registry.entities(
            EntitiesRegistry.Stage.INITIALIZED)
        fs = [(f.__sim_stage__, f, entity) for entity in entities
              for f in entity.__sim_finalizers__]
        fs += [(f.__sim_stage__, f, None)
               for f in static_registry.__finalizers__]
        fs = sort(fs, key=lambda x: x[0])
        for stage, f, entity in fs:
            if entity is None:
                f()
            else:
                f(entity)
        for entity in entities:
            entities_registry.set_stage(
                entity, EntitiesRegistry.Stage.FINISHED)


def check_stop_conditions(entities_registry=None, static_registry=None):
    assert isinstance(entities_registry, EntitiesRegistry)
    assert isinstance(static_registry, StaticRegistry)
    entities = entities_registry.entities(EntitiesRegistry.Stage.INITIALIZED)
    return [(f, e) for e in entities for f in e.check_stop_conditions()] + \
           [(f, None) for f in static_registry.check_stop_conditions()]


def check_death_conditions(entities_registry=None):
    assert isinstance(entities_registry, EntitiesRegistry)
    entities = entities_registry.entities(EntitiesRegistry.Stage.INITIALIZED)
    return [(f, e) for e in entities for f in e.check_death_conditions()]

