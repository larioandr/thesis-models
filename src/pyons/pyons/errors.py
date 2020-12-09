

#######################################################################
# ERRORS
#######################################################################

class MissingFieldError(Exception):
    """
    Exception indicating that the desired field is not set
    (has None value).
    """
    def __init__(self, cls, field):
        self.cls = cls
        self.field = field

    def __str__(self):
        return "Missing field '{cls}.{field}' error".format(
            cls=(self.cls.__name__ if hasattr(self.cls, '__name__')
                 else str(self.cls)),
            field=(self.field.__name__ if hasattr(self.field, '__name__')
                   else str(self.field)))


class NotSupportedError(RuntimeError):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)


class IllegalValueError(RuntimeError):
    """
    Exception indicates that parameter received illegal value.
    """
    def __init__(self, parameter, value, comment=None):
        self.parameter = parameter
        self.value = value
        self.comment = comment

    def __str__(self):
        return "Illegal value {} = '{}'{}".format(
            self.parameter, self.value,
            ": {}".format(self.comment) if self.comment is not None else "")


class SimulationError(Exception):
    """
    Base class for all simulation-related errors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EntityNotFoundError(SimulationError):
    """
    Exception indicating that requested entity is not registered
    within the simulation system.
    """
    def __init__(self, entity):
        super().__init__("Entity not found: '{}'".format(
            entity.__name__ if hasattr(entity, '__name__') else str(entity)))


class EntityAlreadyExists(SimulationError):
    """
    Exception indicates that the given entity is already registered.
    """
    def __init__(self, entity):
        super().__init__("Entity already registered: '{}'".format(
            entity.__name__ if hasattr(entity, '__name__') else str(entity)))


class AmbiguousDefaultHandler(SimulationError):
    """
    Thrown when several default handlers found
    """
    def __init__(self, entity=None):
        super().__init__("Single default handler allowed{}".format(
            "" if entity is None else " per entity, entity='{}'".format(
                (entity.__name__ if hasattr(entity, '__name__')
                 else str(entity)))))


class IllegalManageableFunction(SimulationError):
    """
    Thrown when some manageable function is illegal with the parameters
    provided (e.g. static death condition or unguarded not-default
    event handler in entities)
    """
    def __init__(self, message):
        super().__init__(message)


class EventHandlerNotFound(SimulationError):
    """
    Thrown when fails to find an event handler
    """
    def __init__(self, event, source=None, entity=None):
        super().__init__(
            "handler not found for event={} from source={} at entity={}"
            "".format(event, source, entity))
        self.event = event
        self.source = source
        self.entity = entity


class DispatcherStateError(SimulationError):
    def __init__(self, state, message):
        super().__init__("prohibited operation in kernel state={}{}"
                         "".format(
            state, (": " + message if message is not None else "")))
        self.state = state
        self.message = message


class ResetFromManagedFunction(SimulationError):
    def __init__(self, ctx):
        super().__init__("can not reset kernel from managed function:"
                         "entity={} f={} ftype={}"
                         "".format(ctx.entity, ctx.function, ctx.ftype))
        self.ctx = ctx

