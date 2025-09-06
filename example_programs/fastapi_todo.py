

# ---
# File: /todos/app/api/v1/__init__.py
# ---

from fastapi import APIRouter

from .auth import router as auth_router
from .users import router as users_router
from .priorities import router as priorities_router
from .categories import router as categories_router
from .todos import router as todos_router
from app.core.config import get_config


config = get_config()

router = APIRouter(prefix=config.API_V1_STR)

router.include_router(auth_router)
router.include_router(users_router)
router.include_router(priorities_router)
router.include_router(categories_router)
router.include_router(todos_router)



# ---
# File: /todos/app/api/v1/auth.py
# ---

from fastapi import APIRouter

from app.users.users import fast_api_users
from app.users.auth import auth_backend
from app.schemas import UserRead, UserCreate

router = APIRouter(
    prefix='/auth',
    tags=['Auth']
)


router.include_router(fast_api_users.get_register_router(UserRead, UserCreate))
router.include_router(fast_api_users.get_auth_router(auth_backend))
router.include_router(fast_api_users.get_reset_password_router())
router.include_router(fast_api_users.get_verify_router(UserRead))



# ---
# File: /todos/app/api/v1/categories.py
# ---

from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import conint

from app.core.db import get_async_session
from app.users.users import current_logged_user
from app.dal import db_service, GET_MULTI_DEFAULT_SKIP, GET_MULTI_DEFAULT_LIMIT, MAX_POSTGRES_INTEGER
from app.schemas import CategoryCreate, CategoryRead, CategoryInDB
from app.models.tables import Category, User
from app.utils import exception_handler, get_open_api_response, get_open_api_unauthorized_access_response


router = APIRouter(
    prefix='/categories',
    dependencies=[
        Depends(current_logged_user),
        Depends(get_async_session)
    ],
    tags=['Categories']
)


@router.get(
    '',
    response_model=list[CategoryRead],
    responses={status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response()}
)
async def get_categories(
    skip: conint(ge=0, le=MAX_POSTGRES_INTEGER) = GET_MULTI_DEFAULT_SKIP,  # type: ignore[valid-type]
    limit: conint(ge=0, le=MAX_POSTGRES_INTEGER) = GET_MULTI_DEFAULT_LIMIT,  # type: ignore[valid-type]
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_logged_user)
) -> list[Category]:
    return await db_service.get_categories(
        session,
        created_by_id=user.id,
        skip=skip,
        limit=limit
    )


@router.post(
    '',
    response_model=CategoryRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response(),
        status.HTTP_400_BAD_REQUEST: get_open_api_response(
            {'Trying to add an existing category': 'category name already exists'}
        )
    }
)
@exception_handler
async def add_category(
    category_in: CategoryCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_logged_user)
) -> Category:
    category_in = CategoryInDB(name=category_in.name, created_by_id=user.id)
    return await db_service.add_category(session, category_in=category_in)


@router.delete(
    '/{category_id}',
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response(),
        status.HTTP_403_FORBIDDEN: get_open_api_response(
            {'Trying to delete system or another users category':
             'a user can not delete a category that was not created by him'}
        ),
        status.HTTP_404_NOT_FOUND: get_open_api_response(
            {'Trying to delete non existing category': 'category does not exists'}
        )
    }
)
@exception_handler
async def delete_category(
    category_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_logged_user)
) -> None:
    await db_service.delete_category(session, id_to_delete=category_id, created_by_id=user.id)



# ---
# File: /todos/app/api/v1/priorities.py
# ---

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_async_session
from app.users.users import current_logged_user
from app.dal import db_service
from app.models.tables import Priority
from app.schemas import PriorityRead
from app.utils import get_open_api_unauthorized_access_response


router = APIRouter(
    prefix='/priorities',
    dependencies=[
        Depends(current_logged_user),
        Depends(get_async_session)
    ],
    tags=['Priorities']
)


@router.get(
    '',
    response_model=list[PriorityRead],
    responses={status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response()}
)
async def get_priorities(
    session: AsyncSession = Depends(get_async_session)
) -> Priority:
    return await db_service.get_priorities(session)



# ---
# File: /todos/app/api/v1/todos.py
# ---

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import conint

from app.core.db import get_async_session
from app.users.users import current_logged_user
from app.models.tables import User, Todo
from app.dal import db_service, GET_MULTI_DEFAULT_SKIP, GET_MULTI_DEFAULT_LIMIT, MAX_POSTGRES_INTEGER
from app.schemas import TodoRead, TodoInDB, TodoCreate, TodoUpdate, TodoUpdateInDB
from app.utils import exception_handler, get_open_api_response, get_open_api_unauthorized_access_response


router = APIRouter(
    prefix='/todos',
    dependencies=[
        Depends(current_logged_user),
        Depends(get_async_session)
    ],
    tags=['Todos']
)


@router.get(
    '',
    response_model=list[TodoRead],
    responses={status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response()}
)
async def get_todos(
    skip: conint(ge=0, le=MAX_POSTGRES_INTEGER) = GET_MULTI_DEFAULT_SKIP,  # type: ignore[valid-type]
    limit: conint(ge=0, le=MAX_POSTGRES_INTEGER) = GET_MULTI_DEFAULT_LIMIT,  # type: ignore[valid-type]
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_logged_user)
) -> Todo:
    return await db_service.get_todos(
        session,
        created_by_id=user.id,
        skip=skip,
        limit=limit
    )


@router.post(
    '',
    response_model=TodoRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response(),
        status.HTTP_400_BAD_REQUEST: get_open_api_response(
            {
                'Trying to connect duplicate categories or another users category': 'categories are not valid',
                'Trying to connect non existing priority': 'priority is not valid'
            }
        )

    }
)
@exception_handler
async def add_todo(
    todo_in: TodoCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_logged_user)
) -> Todo:
    todo_in = TodoInDB(
        content=todo_in.content,
        priority_id=todo_in.priority_id,
        categories_ids=todo_in.categories_ids,
        created_by_id=user.id
    )
    return await db_service.add_todo(session, todo_in=todo_in)


@router.put(
    '/{todo_id}',
    response_model=TodoRead,
    responses={
        status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response(),
        status.HTTP_400_BAD_REQUEST: get_open_api_response(
            {
                'Trying to connect duplicate categories or another users category': 'categories are not valid',
                'Trying to connect non existing priority': 'priority is not valid'
            }
        ),
        status.HTTP_403_FORBIDDEN: get_open_api_response(
            {'Trying to update another users todo':
             'a user can not update a todo that was not created by him'}
        ),
        status.HTTP_404_NOT_FOUND: get_open_api_response(
            {'Trying to update non existing todo': 'todo does not exists'}
        )
    }
)
@exception_handler
async def update_todo(
    todo_id: int,
    updated_todo: TodoUpdate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_logged_user)
) -> Todo:
    updated_todo = TodoUpdateInDB(
        id=todo_id,
        content=updated_todo.content,
        priority_id=updated_todo.priority_id,
        categories_ids=updated_todo.categories_ids,
        is_completed=updated_todo.is_completed,
        created_by_id=user.id
    )
    return await db_service.update_todo(session, updated_todo=updated_todo)


@router.delete(
    '/{todo_id}',
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_401_UNAUTHORIZED: get_open_api_unauthorized_access_response(),
        status.HTTP_403_FORBIDDEN: get_open_api_response(
            {'Trying to update another users todo':
             'a user can not update a todo that was not created by him'}
        ),
        status.HTTP_404_NOT_FOUND: get_open_api_response(
            {'Trying to update non existing todo': 'todo does not exists'}
        )
    }
)
@exception_handler
async def delete_todo(
    todo_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_logged_user)
) -> None:
    await db_service.delete_todo(session, id_to_delete=todo_id, created_by_id=user.id)



# ---
# File: /todos/app/api/v1/users.py
# ---

from fastapi import APIRouter

from app.users.users import fast_api_users
from app.schemas import UserRead, UserUpdate

router = APIRouter(
    prefix='/users',
    tags=['Users']
)

router.include_router(fast_api_users.get_users_router(UserRead, UserUpdate))



# ---
# File: /todos/app/api/__init__.py
# ---

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.v1 import router as v1_router


router = APIRouter()
router.include_router(v1_router)
router.include_router(health_router)



# ---
# File: /todos/app/api/health.py
# ---

import socket
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import APIRouter, Depends, status, Response

from app.core.db import get_async_session

router = APIRouter(
    prefix='/health',
    dependencies=[Depends(get_async_session)],
    tags=['Health']
)


@router.get(
    '',
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                'description': 'Database connection is unavailable',
            }
        }
)
async def health(
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    try:
        #  SELECT 1 is a simple SQL query used to test database connectivity
        await asyncio.wait_for(session.execute(select(1)), timeout=1)
    except (asyncio.TimeoutError, socket.gaierror):
        #  socket.gaierror exception is raised when there is an error resolving a hostname. In this case,
        #  it is being used to handle network-related errors that may occur when attempting to connect to the database
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response(status_code=204)



# ---
# File: /todos/app/core/__init__.py
# ---




# ---
# File: /todos/app/core/config.py
# ---

from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseSettings, PostgresDsn, AnyHttpUrl, validator, SecretStr, EmailStr


class Settings(BaseSettings):
    PROJECT_NAME: str = 'Todos API'
    API_V1_STR: str = '/api/v1'
    JWT_SECRET_KEY: SecretStr

    # 60 seconds by 60 minutes (1 hour) and then by 12 (for 12 hours total)
    JWT_LIFETIME_SECONDS: int = 60 * 60 * 12

    # CORS_ORIGINS is a string of ';' separated origins.
    # e.g:  'http://localhost:8080;http://localhost:3000'
    CORS_ORIGINS: list[AnyHttpUrl]

    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_URI: Optional[PostgresDsn] = None

    @validator('POSTGRES_URI', pre=True)
    def assemble_db_connection(cls, _: str, values: dict[str, Any]) -> str:
        postgres_password: SecretStr = values.get('POSTGRES_PASSWORD', SecretStr(''))
        return PostgresDsn.build(
            scheme='postgresql+asyncpg',
            user=values.get('POSTGRES_USER'),
            password=postgres_password.get_secret_value(),
            host=values.get('POSTGRES_HOST'),
            path=f'/{values.get("POSTGRES_DB")}',
        )

    SMTP_TLS: bool = True
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: Optional[int] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[SecretStr] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None

    @validator('EMAILS_FROM_NAME')
    def get_project_name(cls, v: Optional[str], values: dict[str, Any]) -> str:
        if not v:
            return values['PROJECT_NAME']
        return v

    EMAIL_TEMPLATES_DIR: str = './todos/app/email-templates'
    EMAILS_ENABLED: bool = False

    @validator('EMAILS_ENABLED', pre=True)
    def get_emails_enabled(cls, _: bool, values: dict[str, Any]) -> bool:
        return all([
            values.get('SMTP_HOST'),
            values.get('SMTP_PORT'),
            values.get('EMAILS_FROM_EMAIL')
        ])

    # 60 seconds by 60 minutes (1 hour) and then by 12 (for 12 hours total)
    RESET_PASSWORD_TOKEN_LIFETIME_SECONDS: int = 60 * 60 * 12
    VERIFY_TOKEN_LIFETIME_SECONDS: int = 60 * 60 * 12

    FRONT_END_BASE_URL: AnyHttpUrl

    class Config:
        env_file = '.env'
        case_sensitive = True

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name == 'CORS_ORIGINS':
                return [origin for origin in raw_val.split(';')]
            # The following line is ignored by mypy because:
            # error: Type'[Config]' has no attribute 'json_loads',
            # even though it is like the documentation: https://docs.pydantic.dev/latest/usage/settings/
            return cls.json_loads(raw_val)  # type: ignore[attr-defined]


@lru_cache()
def get_config() -> Settings:
    # TODO: remove 'type: ignore[call-arg]' once https://github.com/pydantic/pydantic/issues/3072 is closed
    return Settings()  # type: ignore[call-arg]



# ---
# File: /todos/app/core/db.py
# ---

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_config

config = get_config()

engine: AsyncEngine = create_async_engine(config.POSTGRES_URI, echo=True)

Session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, Any]:
    async with Session() as session:
        yield session



# ---
# File: /todos/app/dal/__init__.py
# ---

from .db_service import db_service
from .constants import MAX_POSTGRES_INTEGER, GET_MULTI_DEFAULT_SKIP, GET_MULTI_DEFAULT_LIMIT



# ---
# File: /todos/app/dal/constants.py
# ---

from typing import Final


# maximum value that can be represented by a 32-bit signed integer.
# if trying to send a bigger value than that in queries (like offset or limit)
# the database throws an error - OverflowError: value out of int32 range
MAX_POSTGRES_INTEGER: Final[int] = (2 ** 31) - 1

GET_MULTI_DEFAULT_SKIP: Final[int] = 0
GET_MULTI_DEFAULT_LIMIT: Final[int] = 100



# ---
# File: /todos/app/dal/db_repo.py
# ---

from typing import Optional, Type, TypeVar, Union, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.models.base import Base
from app.schemas.base import BaseInDB, BaseUpdateInDB
from app.dal.constants import GET_MULTI_DEFAULT_SKIP


ModelType = TypeVar('ModelType', bound=Base)
InDBSchemaType = TypeVar('InDBSchemaType', bound=BaseInDB)
UpdateSchemaType = TypeVar('UpdateSchemaType', bound=BaseUpdateInDB)


class DBRepo:

    def __init__(self) -> None:
        ...

    async def get(  # type: ignore[no-untyped-def]
        self,
        session: AsyncSession,
        *,
        table_model: Type[ModelType],
        query_filter=None  # type: ignore
    ) -> Union[Optional[ModelType]]:
        query = select(table_model)
        if query_filter is not None:
            query = query.filter(query_filter)
        result = await session.execute(query)
        return result.scalars().first()

    async def get_multi(    # type: ignore[no-untyped-def]
        self,
        session: AsyncSession,
        *,
        table_model: Type[ModelType],
        query_filter=None,
        skip: int = GET_MULTI_DEFAULT_SKIP,
        limit: Optional[int] = None
    ) -> list[ModelType]:
        query = select(table_model)
        if query_filter is not None:
            query = query.filter(query_filter)
        query = query.offset(skip)
        if limit is not None:
            query = query.limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

    async def create(
        self,
        session: AsyncSession,
        *,
        obj_to_create: InDBSchemaType
    ) -> ModelType:
        db_obj: ModelType = obj_to_create.to_orm()
        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)
        return db_obj

    async def update(
        self,
        session: AsyncSession,
        *,
        updated_obj: UpdateSchemaType,
        db_obj_to_update: Optional[ModelType] = None
    ) -> Optional[ModelType]:
        existing_obj_to_update: Optional[ModelType] = db_obj_to_update or await self.get(
            session,
            table_model=updated_obj.Config.orm_model,
            query_filter=updated_obj.Config.orm_model.id == updated_obj.id
        )
        if existing_obj_to_update:
            existing_obj_to_update_data = existing_obj_to_update.dict()
            updated_data: dict[str, Any] = updated_obj.to_orm().dict()
            for field in existing_obj_to_update_data:
                if field in updated_data:
                    setattr(existing_obj_to_update, field, updated_data[field])
            session.add(existing_obj_to_update)
            await session.commit()
            await session.refresh(existing_obj_to_update)
        return existing_obj_to_update

    async def delete(
        self,
        session: AsyncSession,
        *,
        table_model: Type[ModelType],
        id_to_delete: int
    ) -> None:
        query = delete(table_model).where(table_model.id == id_to_delete)
        await session.execute(query)
        await session.commit()



# ---
# File: /todos/app/dal/db_service.py
# ---

import uuid
from typing import Optional

from sqlalchemy import or_, and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.dal.db_repo import DBRepo
from app.dal.constants import GET_MULTI_DEFAULT_SKIP, GET_MULTI_DEFAULT_LIMIT
from app.models.tables import Priority, Category, Todo
from app.schemas import CategoryInDB, TodoInDB, TodoUpdateInDB
from app.http_exceptions import ResourceNotExists, UserNotAllowed, ResourceAlreadyExists


class DBService:

    def __init__(self) -> None:
        self._repo = DBRepo()

    async def _validate_todo_categories(
        self,
        session: AsyncSession,
        *,
        todo_categories_ids: list[int],
        created_by_id: uuid.UUID
    ) -> bool:
        # validates that the todo categories are valid to the user + no duplications
        default_categories_filter = Category.created_by_id.is_(None)
        user_categories_filter = Category.created_by_id == created_by_id
        valid_categories_filter = or_(default_categories_filter, user_categories_filter)
        todo_categories_ids_filter = Category.id.in_(todo_categories_ids)

        categories_from_db: list[Category] = await self._repo.get_multi(
            session,
            table_model=Category,
            query_filter=and_(valid_categories_filter, todo_categories_ids_filter)
        )
        are_categories_valid: bool = len(todo_categories_ids) == len(categories_from_db)
        return are_categories_valid

    async def get_priorities(self, session: AsyncSession) -> list[Priority]:
        return await self._repo.get_multi(session, table_model=Priority)

    async def get_categories(
        self,
        session: AsyncSession,
        *,
        created_by_id: uuid.UUID,
        skip: int = GET_MULTI_DEFAULT_SKIP,
        limit: int = GET_MULTI_DEFAULT_LIMIT
    ) -> list[Category]:
        default_categories_filter = Category.created_by_id.is_(None)
        user_categories_filter = Category.created_by_id == created_by_id
        query_filter = or_(user_categories_filter, default_categories_filter)
        return await self._repo.get_multi(
            session,
            table_model=Category,
            query_filter=query_filter,
            limit=limit,
            skip=skip
        )

    async def add_category(
        self,
        session: AsyncSession,
        *,
        category_in: CategoryInDB
    ) -> Category:
        users_categories: list[Category] = await self.get_categories(
            session,
            created_by_id=category_in.created_by_id)
        users_categories_names: list[str] = [c.name for c in users_categories]
        if category_in.name in users_categories_names:
            raise ResourceAlreadyExists(resource='category name')
        return await self._repo.create(session, obj_to_create=category_in)

    async def delete_category(
        self,
        session: AsyncSession,
        *,
        id_to_delete: int,
        created_by_id: uuid.UUID
    ) -> None:
        category_to_delete: Optional[Category] = await self._repo.get(
            session,
            table_model=Category,
            query_filter=Category.id == id_to_delete
        )
        if not category_to_delete:
            raise ResourceNotExists(resource='category')
        if category_to_delete.created_by_id != created_by_id:
            raise UserNotAllowed('a user can not delete a category that was not created by him')
        await self._repo.delete(session, table_model=Category, id_to_delete=id_to_delete)

    async def get_todos(
        self,
        session: AsyncSession,
        *,
        created_by_id: uuid.UUID,
        skip: int = GET_MULTI_DEFAULT_SKIP,
        limit: int = GET_MULTI_DEFAULT_LIMIT
    ) -> list[Todo]:
        return await self._repo.get_multi(
            session,
            table_model=Todo,
            query_filter=Todo.created_by_id == created_by_id,
            skip=skip,
            limit=limit
        )

    async def add_todo(
        self,
        session: AsyncSession,
        *,
        todo_in: TodoInDB
    ) -> Todo:
        if await self._validate_todo_categories(
            session,
            todo_categories_ids=todo_in.categories_ids,
            created_by_id=todo_in.created_by_id
        ):
            try:
                return await self._repo.create(session, obj_to_create=todo_in)
            except IntegrityError:
                raise ValueError('priority is not valid')
        raise ValueError('categories are not valid')

    async def update_todo(
        self,
        session: AsyncSession,
        *,
        updated_todo: TodoUpdateInDB
    ) -> Todo:
        todo_to_update: Optional[Todo] = await self._repo.get(
            session,
            table_model=Todo,
            query_filter=Todo.id == updated_todo.id
        )
        if not todo_to_update:
            raise ResourceNotExists(resource='todo')
        if not todo_to_update.created_by_id == updated_todo.created_by_id:
            raise UserNotAllowed('a user can not update a todo that was not created by him')
        if await self._validate_todo_categories(
            session,
            todo_categories_ids=updated_todo.categories_ids,
            created_by_id=updated_todo.created_by_id
        ):
            try:
                todo_updated_obj: Optional[Todo] = await self._repo.update(
                    session,
                    updated_obj=updated_todo,
                    db_obj_to_update=todo_to_update
                )
                if todo_updated_obj:
                    return todo_updated_obj
                raise ResourceNotExists(resource='todo')
            except IntegrityError:
                raise ValueError('priority is not valid')
        raise ValueError('categories are not valid')

    async def delete_todo(
        self,
        session: AsyncSession,
        *,
        id_to_delete: int,
        created_by_id: uuid.UUID
    ) -> None:
        todo_to_delete: Optional[Todo] = await self._repo.get(
            session,
            table_model=Todo,
            query_filter=Todo.id == id_to_delete
        )
        if not todo_to_delete:
            raise ResourceNotExists(resource='todo')
        if todo_to_delete.created_by_id != created_by_id:
            raise UserNotAllowed('a user can not delete a todo that was not created by him')
        await self._repo.delete(session, table_model=Todo, id_to_delete=id_to_delete)


db_service = DBService()



# ---
# File: /todos/app/http_exceptions/__init__.py
# ---

from .resource_already_exists import ResourceAlreadyExists
from .resource_not_exists import ResourceNotExists
from .user_not_allowed import UserNotAllowed



# ---
# File: /todos/app/http_exceptions/resource_already_exists.py
# ---

class ResourceAlreadyExists(Exception):
    def __init__(self, *, resource: str):
        self.msg = f'{resource} already exists'
        super().__init__(self.msg)



# ---
# File: /todos/app/http_exceptions/resource_not_exists.py
# ---

class ResourceNotExists(Exception):
    def __init__(self, *, resource: str):
        self.msg = f'{resource} does not exist'
        super().__init__(self.msg)



# ---
# File: /todos/app/http_exceptions/user_not_allowed.py
# ---

class UserNotAllowed(Exception):
    ...



# ---
# File: /todos/app/models/__init__.py
# ---





# ---
# File: /todos/app/models/base.py
# ---

from typing import Any

import humps
from sqlalchemy.ext.declarative import declared_attr, as_declarative
from sqlalchemy import inspect


@as_declarative()
class Base:
    __name__: str

    @declared_attr
    # The following line is ignored by pylint even though it is like the documentation:
    # https://docs.sqlalchemy.org/en/14/orm/extensions/mypy.html#using-declared-attr-and-declarative-mixins
    def __tablename__(cls) -> str:  # pylint: disable=no-self-argument
        return humps.depascalize(cls.__name__)

    def dict(self) -> dict[str, Any]:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

    def __repr__(self) -> str:
        columns = [f'{col}: {getattr(self, col)}' for col in self.dict()]
        return f'{self.__class__.__name__}({", ".join(columns)})'

    def __str__(self) -> str:
        return self.__repr__()



# ---
# File: /todos/app/models/tables.py
# ---

from typing import Union

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy import GUID
from sqlalchemy import Column, ForeignKey, Text, String, BigInteger, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship, RelationshipProperty

from app.models.base import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    ...


class Priority(Base):
    id = Column(BigInteger(), primary_key=True, autoincrement=True)
    name = Column(String(15), nullable=False, unique=True)


class Category(Base):
    id = Column(BigInteger(), primary_key=True, autoincrement=True)
    name = Column(Text(), nullable=False)
    # Default categories are those where created_by_id is NULL,
    # indicating they are created by the system and are applicable to all users
    created_by_id = Column(GUID, ForeignKey('user.id'))

    __table_args__ = (
        UniqueConstraint('name', 'created_by_id', name='unique_category'),
    )

    todos: RelationshipProperty = relationship(
        'Todo',
        secondary='todo_category',
        back_populates='categories',
        viewonly=True
    )


class Todo(Base):
    id = Column(BigInteger(), primary_key=True, autoincrement=True)
    is_completed = Column(Boolean(), nullable=False, default=False)
    content = Column(Text(), nullable=False)
    created_by_id = Column(GUID, ForeignKey('user.id'), nullable=False)
    priority_id = Column(BigInteger(), ForeignKey('priority.id'), nullable=False)

    priority: RelationshipProperty = relationship('Priority', lazy='selectin')
    categories: RelationshipProperty = relationship(
        'Category',
        secondary='todo_category',
        back_populates='todos',
        lazy='selectin',
        viewonly=True
    )
    # just for adding todos_categories when adding a todo
    todos_categories: RelationshipProperty = relationship(
        'TodoCategory',
        lazy='selectin',
        cascade='all, delete-orphan'
    )

    def dict(self) -> dict:
        # adding todos_categories field to dict()
        # just update usage only
        todo_dict: dict[str, Union[int, str, bool]] = super().dict()
        todo_dict['todos_categories'] = self.todos_categories  # type: ignore[assignment]
        return todo_dict


class TodoCategory(Base):
    todo_id = Column(
        BigInteger(),
        ForeignKey('todo.id', ondelete='CASCADE'), primary_key=True
    )
    category_id = Column(
        BigInteger(),
        ForeignKey('category.id', ondelete='CASCADE'), primary_key=True
    )



# ---
# File: /todos/app/schemas/__init__.py
# ---

# Read - properties to return to client
# Create - properties to receive on item creation
# Update - properties to receive on item update
# InDB -  properties stored in DB

from .base import BaseInDB
from .user import UserRead, UserCreate, UserUpdate
from .priority import PriorityRead
from .category import CategoryRead, CategoryCreate, CategoryInDB
from .todo import TodoRead, TodoCreate, TodoInDB, TodoUpdate, TodoUpdateInDB



# ---
# File: /todos/app/schemas/base.py
# ---

from pydantic import BaseModel
from typing import Type, Optional

from app.models.base import Base


class BaseInDB(BaseModel):
    # base schema for every schema that stored in DB.
    # provides a default method for converting
    # Pydantic objects to SQLAlchemy objects
    class Config:
        orm_model: Optional[Type[Base]] = None

    def to_orm(self) -> Base:
        if not self.Config.orm_model:
            raise AttributeError('Class has not defined Config.orm_model')
        return self.Config.orm_model(**dict(self))  # pylint: disable=not-callable


class BaseUpdateInDB(BaseInDB):
    id: int



# ---
# File: /todos/app/schemas/category.py
# ---

import uuid
from typing import Optional

from pydantic import BaseModel

from app.schemas.base import BaseInDB
from app.models.tables import Category


class CategoryCreate(BaseModel):
    name: str


class CategoryRead(CategoryCreate):
    id: int

    class Config:
        orm_mode = True


class CategoryInDB(BaseInDB, CategoryCreate):
    created_by_id: Optional[uuid.UUID]

    class Config(BaseInDB.Config):
        orm_model = Category



# ---
# File: /todos/app/schemas/priority.py
# ---

from pydantic import BaseModel


class PriorityRead(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True



# ---
# File: /todos/app/schemas/todo.py
# ---

import uuid

from pydantic import BaseModel

from app.schemas.base import BaseInDB, BaseUpdateInDB
from app.schemas.priority import PriorityRead
from app.schemas.category import CategoryRead
from app.models.tables import Todo, TodoCategory


class TodoBase(BaseModel):
    content: str


class TodoRead(TodoBase):
    id: int
    is_completed: bool
    priority: PriorityRead
    categories: list[CategoryRead]

    class Config:
        orm_mode = True


class TodoCreate(TodoBase):
    priority_id: int
    categories_ids: list[int]


class TodoInDB(BaseInDB, TodoCreate):
    created_by_id: uuid.UUID
    priority_id: int

    class Config(BaseInDB.Config):
        orm_model = Todo

    def to_orm(self) -> Todo:
        # converts categories_ids to todos_categories
        orm_data = dict(self)
        categories_ids = orm_data.pop('categories_ids')
        todo_orm = self.Config.orm_model(**orm_data)
        todo_orm.todos_categories = [TodoCategory(category_id=c_id) for c_id in categories_ids]
        return todo_orm


class TodoUpdate(TodoCreate):
    is_completed: bool


class TodoUpdateInDB(BaseUpdateInDB, TodoInDB):
    is_completed: bool



# ---
# File: /todos/app/schemas/user.py
# ---

import uuid

from pydantic import EmailStr
from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    email: EmailStr
    is_superuser: bool


class UserCreate(schemas.BaseUserCreate):
    email: EmailStr
    password: str


class UserUpdate(schemas.BaseUserUpdate):
    password: str



# ---
# File: /todos/app/users/__init__.py
# ---




# ---
# File: /todos/app/users/auth.py
# ---

from fastapi_users.authentication import AuthenticationBackend, BearerTransport

from app.users.security import get_jwt_strategy
from app.core.config import get_config


config = get_config()


bearer_transport = BearerTransport(tokenUrl=f'{config.API_V1_STR}/auth/login')

auth_backend = AuthenticationBackend(
    name='jwt',
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)



# ---
# File: /todos/app/users/manager.py
# ---

import uuid
from typing import Optional
import logging

from pydantic import SecretStr
from fastapi import Request
from fastapi_users import BaseUserManager, UUIDIDMixin

from app.core.config import get_config
from app.models.tables import User
from app.utils import send_reset_password_email, send_account_verification_email


config = get_config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret: SecretStr = config.JWT_SECRET_KEY
    reset_password_token_lifetime_seconds: int = config.RESET_PASSWORD_TOKEN_LIFETIME_SECONDS
    verification_token_secret: SecretStr = config.JWT_SECRET_KEY
    verification_token_lifetime_seconds: int = config.VERIFY_TOKEN_LIFETIME_SECONDS

    async def on_after_forgot_password(
            self,
            user: User,
            token: str,
            request: Optional[Request] = None
    ) -> None:
        send_reset_password_email(email_to=user.email, token=token)
        logger.info('sent reset password email to %s', user.email)

    async def on_after_request_verify(
            self,
            user: User,
            token: str,
            request: Optional[Request] = None
    ) -> None:
        send_account_verification_email(email_to=user.email, token=token)
        logger.info('sent account verification email to %s', user.email)



# ---
# File: /todos/app/users/security.py
# ---

from typing import Optional, Final, Union

from fastapi_users.authentication import JWTStrategy
from fastapi_users.jwt import SecretType, generate_jwt

from app.core.config import get_config
from app.models.tables import User


config = get_config()

JWT_HASHING_ALGORITHM: Final[str] = 'HS256'


class TodosJWTStrategy(JWTStrategy):
    def __init__(
            self,
            secret: SecretType,
            lifetime_seconds: Optional[int],
            token_audience: Optional[list[str]] = None,
            algorithm: str = JWT_HASHING_ALGORITHM,
            public_key: Optional[SecretType] = None,
    ):
        if token_audience is None:
            token_audience = ['fastapi-users:auth', 'fastapi-users:verify']
        super().__init__(secret=secret, lifetime_seconds=lifetime_seconds,
                         token_audience=token_audience, algorithm=algorithm,
                         public_key=public_key)

    async def write_token(self, user: User) -> str:
        data = self.generate_jwt_data(user)
        return generate_jwt(data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm)

    def generate_jwt_data(self, user: User) -> dict[str, Union[str, list[str], bool]]:
        return dict(user_id=str(user.id),
                    aud=self.token_audience,
                    email=user.email,
                    isSuperuser=user.is_superuser)


def get_jwt_strategy() -> JWTStrategy:
    return TodosJWTStrategy(
        secret=config.JWT_SECRET_KEY,
        lifetime_seconds=config.JWT_LIFETIME_SECONDS
    )



# ---
# File: /todos/app/users/users.py
# ---

import uuid
from typing import Any
from collections.abc import AsyncGenerator

from fastapi import Depends
from fastapi_users import FastAPIUsers
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.users.auth import auth_backend
from app.users.manager import UserManager
from app.models.tables import User
from app.core.db import get_async_session


async def get_user_db(session: AsyncSession = Depends(get_async_session)) -> \
        AsyncGenerator[SQLAlchemyUserDatabase, User]:
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)) ->\
        AsyncGenerator[UserManager, Any]:
    yield UserManager(user_db)


fast_api_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_logged_user = fast_api_users.current_user(active=True, verified=False, superuser=False)



# ---
# File: /todos/app/utils/__init__.py
# ---

from .emails import send_reset_password_email, send_account_verification_email
from .exceptions import exception_handler
from.open_api import get_open_api_response, get_open_api_unauthorized_access_response



# ---
# File: /todos/app/utils/emails.py
# ---

from typing import Any, Optional
import logging

from emails import Message
from emails.template import JinjaTemplate

from app.core.config import get_config


config = get_config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_email(
    *,
    email_to: str,
    environment: Optional[dict[str, Any]],
    subject_template: str = "",
    html_template: str = "",
) -> None:
    if not config.EMAILS_ENABLED:
        raise RuntimeError('no configuration provided for email variables')
    if not environment:
        environment = {}
    message = Message(
        subject=JinjaTemplate(subject_template),
        html=JinjaTemplate(html_template),
        mail_from=(config.EMAILS_FROM_NAME, config.EMAILS_FROM_EMAIL),
    )
    smtp_options = {'host': config.SMTP_HOST, 'port': config.SMTP_PORT}
    if config.SMTP_TLS:
        smtp_options['tls'] = True
    if config.SMTP_USER:
        smtp_options['user'] = config.SMTP_USER
    if config.SMTP_PASSWORD:
        smtp_options['password'] = config.SMTP_PASSWORD.get_secret_value()
    res = message.send(to=email_to, render=environment, smtp=smtp_options)
    logger.info('send email result %s', res)


def send_reset_password_email(*, email_to: str, token: str) -> None:
    subject = f'{config.PROJECT_NAME} - Password recovery for email {email_to}'
    with open(f'{config.EMAIL_TEMPLATES_DIR}/reset_password.html', 'r', encoding='utf-8') as f:
        template_str = f.read()
    link = f'{config.FRONT_END_BASE_URL}/reset-password?token={token}'
    send_email(
        email_to=email_to,
        subject_template=subject,
        html_template=template_str,
        environment={
            'project_name': config.PROJECT_NAME,
            'email': email_to,
            'link': link,
            # dividing by 3600 to get the number of hours from the number of seconds
            'expire_hours': config.RESET_PASSWORD_TOKEN_LIFETIME_SECONDS / 3600,
        }
    )


def send_account_verification_email(*, email_to: str, token: str) -> None:
    subject = f'{config.PROJECT_NAME} - Account verification for email {email_to}'
    with open(f'{config.EMAIL_TEMPLATES_DIR}/account_verification.html', 'r', encoding='utf-8') as f:
        template_str = f.read()
    link = f'{config.FRONT_END_BASE_URL}/verify-account?token={token}'
    send_email(
        email_to=email_to,
        subject_template=subject,
        html_template=template_str,
        environment={
            'project_name': config.PROJECT_NAME,
            'email': email_to,
            'link': link,
            # dividing by 3600 to get the number of hours from the number of seconds
            'expire_hours': config.VERIFY_TOKEN_LIFETIME_SECONDS / 3600,
        }
    )



# ---
# File: /todos/app/utils/exceptions.py
# ---

from typing import Callable, Any, Type
from functools import wraps

from fastapi import status, HTTPException

from app.http_exceptions import ResourceNotExists, UserNotAllowed, ResourceAlreadyExists


def exception_handler(f: Callable) -> Any:
    exception_map: dict[Type[Exception], int] = {
        ValueError: status.HTTP_400_BAD_REQUEST,
        UserNotAllowed: status.HTTP_403_FORBIDDEN,
        ResourceNotExists: status.HTTP_404_NOT_FOUND,
        ResourceAlreadyExists: status.HTTP_409_CONFLICT,
    }

    exceptions: tuple[Type[Exception], ...] = tuple(exception_map.keys())

    @wraps(f)
    async def decorated(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        try:
            return await f(*args, **kwargs)
        except exceptions as err:
            exception_cls = type(err)
            status_code = exception_map[exception_cls]
            raise HTTPException(status_code=status_code, detail=str(err))
    return decorated



# ---
# File: /todos/app/utils/open_api.py
# ---

from typing import Union

from fastapi_users.openapi import OpenAPIResponseType
from fastapi_users.router.common import ErrorModel


def get_open_api_response(examples_res_details: dict[str, str]) -> OpenAPIResponseType:
    examples: dict[str, dict[str, Union[str, dict[str, str]]]] = {}
    for example, res_detail in examples_res_details.items():
        examples[example] = {
            'summary': example,
            'value': {'detail': res_detail}
        }
    return {
        # https://fastapi.tiangolo.com/advanced/additional-responses/#additional-response-with-model
        'model': ErrorModel,  # type: ignore[dict-item]
        'content': {
            'application/json': {
                'examples': examples
            }
        }
    }


def get_open_api_unauthorized_access_response() -> OpenAPIResponseType:
    return get_open_api_response({'Unauthorized access': 'Unauthorized'})



# ---
# File: /todos/app/__init__.py
# ---




# ---
# File: /todos/app/main.py
# ---

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.core.config import get_config

config = get_config()

app = FastAPI(
    title=config.PROJECT_NAME,
    openapi_url=f'{config.API_V1_STR}/openapi.json'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router)


if __name__ == '__main__':
    uvicorn.run(app)



# ---
# File: /todos/migrations/versions/71139a54084d_first_migration.py
# ---

"""first_migration

Revision ID: 71139a54084d
Revises: 
Create Date: 2023-05-04 12:08:41.445166

"""
import fastapi_users_db_sqlalchemy
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '71139a54084d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('priority',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=15), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('user',
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('hashed_password', sa.String(length=1024), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('is_superuser', sa.Boolean(), nullable=False),
    sa.Column('is_verified', sa.Boolean(), nullable=False),
    sa.Column('id', fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_email'), 'user', ['email'], unique=True)
    op.create_table('category',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('created_by_id', fastapi_users_db_sqlalchemy.generics.GUID(), nullable=True),
    sa.ForeignKeyConstraint(['created_by_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'created_by_id', name='unique_category')
    )
    op.create_table('todo',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('is_completed', sa.Boolean(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('created_by_id', fastapi_users_db_sqlalchemy.generics.GUID(), nullable=False),
    sa.Column('priority_id', sa.BigInteger(), nullable=False),
    sa.ForeignKeyConstraint(['created_by_id'], ['user.id'], ),
    sa.ForeignKeyConstraint(['priority_id'], ['priority.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('todo_category',
    sa.Column('todo_id', sa.BigInteger(), nullable=False),
    sa.Column('category_id', sa.BigInteger(), nullable=False),
    sa.ForeignKeyConstraint(['category_id'], ['category.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['todo_id'], ['todo.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('todo_id', 'category_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('todo_category')
    op.drop_table('todo')
    op.drop_table('category')
    op.drop_index(op.f('ix_user_email'), table_name='user')
    op.drop_table('user')
    op.drop_table('priority')
    # ### end Alembic commands ###



# ---
# File: /todos/migrations/env.py
# ---

from logging.config import fileConfig

import asyncio
from sqlalchemy import pool, engine_from_config
from sqlalchemy.engine.base import Connection
from sqlalchemy.ext.asyncio import AsyncEngine
from alembic import context

from app.models.tables import Base
from app.core.config import get_config

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
config.set_main_option('sqlalchemy.url', get_config().POSTGRES_URI)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option('sqlalchemy.url')
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
        compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = AsyncEngine(
        engine_from_config(  # type: ignore[arg-type]
            config.get_section(config.config_ini_section),
            prefix='sqlalchemy.',
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())



# ---
# File: /todos/scripts/initial_data.py
# ---

from typing import Final
import json
import logging

import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import Session
from app.models.tables import Priority, Category


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INITIAL_DATA_FILE_PATH: Final[str] = 'todos/scripts/initial_data.json'


async def initiate_data(session: AsyncSession) -> None:
    with open(INITIAL_DATA_FILE_PATH, 'r') as f:
        initial_data_dict: dict[str, list[str]] = json.load(f)

    # initiate priorities
    initial_priorities_names: list[str] = initial_data_dict['priorities_names']
    priorities_result = await session.execute(
        (select(Priority).filter(Priority.name.in_(initial_priorities_names)))
    )
    priorities_from_db: list[Priority] = priorities_result.scalars().all()
    if not priorities_from_db:
        priorities: list[Priority] = [
            Priority(name=priority_name) for priority_name in initial_priorities_names
        ]
        session.add_all(priorities)

    # initiate categories
    initial_categories_names: list[str] = initial_data_dict['categories_names']
    categories_result = await session.execute(
        (select(Category).filter(Category.name.in_(initial_categories_names)))
    )
    categories_from_db: list[Category] = categories_result.scalars().all()
    if not categories_from_db:
        categories: list[Category] = [
            Category(name=category_name, created_by_id=None) for category_name in initial_categories_names
        ]
        session.add_all(categories)
    await session.commit()


async def main() -> None:
    logger.info('Creating initial data')
    async with Session() as session:
        await initiate_data(session)
    logger.info('Initial data created')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())



# ---
# File: /todos/tests/api/__init__.py
# ---




# ---
# File: /todos/tests/api/test_categories.py
# ---

import pytest
from pytest_lazyfixture import lazy_fixture
from httpx import AsyncClient

from tests.conftest_utils import get_tests_data
from app.core.config import get_config


config = get_config()


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, status_code, res_body', [
    (None, 401, {'detail': 'Unauthorized'}),
    (
        lazy_fixture('user_token_headers'),
        200,
        get_tests_data()['categories'] + get_tests_data()['users'][0]['categories']
    )
], ids=['unauthorized access', 'authorized access'])
async def test_get_categories(
    client: AsyncClient,
    headers,
    status_code,
    res_body
):
    res = await client.get(f'{config.API_V1_STR}/categories', headers=headers)
    assert res.status_code == status_code
    assert res.json() == res_body


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, data, status_code, res_body', [
    (None, {'name': 'Work'}, 401, {'detail': 'Unauthorized'}),
    (lazy_fixture('user_token_headers'), {'name': 'Personal'}, 409, {'detail': 'category name already exists'}),
    (lazy_fixture('user_token_headers'), {'name': 'Chess'}, 409, {'detail': 'category name already exists'}),
    (lazy_fixture('user_token_headers'), {'name': 'Nintendo'}, 201, {'name': 'Nintendo', 'id': 5})
], ids=[
    'unauthorized access',
    'authorized access default existing category',
    'authorized access another users existing category',
    'authorized access non existing category'
])
async def test_add_category(
    client: AsyncClient,
    headers,
    data,
    status_code,
    res_body
):
    res = await client.post(f'{config.API_V1_STR}/categories', headers=headers, json=data)
    assert res.status_code == status_code
    assert res.json() == res_body


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, category_id, status_code, res_body', [
    (None, 1, 401, {'detail': 'Unauthorized'}),
    (lazy_fixture('user_token_headers'), 5, 404, {'detail': 'category does not exist'}),
    (
        lazy_fixture('user_token_headers'),
        1,
        403,
        {'detail': 'a user can not delete a category that was not created by him'}
    ),
    (
        lazy_fixture('user_token_headers'),
        4,
        403,
        {'detail': 'a user can not delete a category that was not created by him'}
    )
], ids=[
    'unauthorized access',
    'authorized access non existing category',
    'authorized access default existing category',
    'authorized access another users existing category'
])
async def test_delete_category_failure(
    client: AsyncClient,
    headers,
    category_id,
    status_code,
    res_body
):
    res = await client.delete(f'{config.API_V1_STR}/categories/{category_id}', headers=headers)
    assert res.status_code == status_code
    assert res.json() == res_body


@pytest.mark.asyncio
async def test_delete_category_success(
    client: AsyncClient,
    user_token_headers: dict[str, str]
):
    res = await client.delete(f'{config.API_V1_STR}/categories/3', headers=user_token_headers)
    assert res.status_code == 204
    assert len(res.content) == 0



# ---
# File: /todos/tests/api/test_priorities.py
# ---

import pytest
from pytest_lazyfixture import lazy_fixture
from httpx import AsyncClient

from tests.conftest_utils import get_tests_data
from app.core.config import get_config


config = get_config()


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, status_code, res_body', [
    (None, 401, {'detail': 'Unauthorized'}),
    (lazy_fixture('user_token_headers'), 200, get_tests_data()['priorities'])
], ids=['unauthorized access', 'authorized access'])
async def test_get_priorities(
    client: AsyncClient,
    headers,
    status_code,
    res_body
):
    res = await client.get(f'{config.API_V1_STR}/priorities', headers=headers)
    assert res.status_code == status_code
    assert res.json() == res_body



# ---
# File: /todos/tests/api/test_todos.py
# ---

from typing import Final

import pytest
from pytest_lazyfixture import lazy_fixture
from httpx import AsyncClient

from tests.conftest_utils import get_tests_data
from app.core.config import get_config


config = get_config()

API_TODOS_PREFIX: Final[str] = f'{config.API_V1_STR}/todos'


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, status_code, res_body', [
    (None, 401, {'detail': 'Unauthorized'}),
    (
        lazy_fixture('user_token_headers'),
        200,
        get_tests_data()['users'][0]['todos']
    )
], ids=['unauthorized access', 'authorized access'])
async def test_get_todos(
    client: AsyncClient,
    headers,
    status_code,
    res_body
):
    res = await client.get(API_TODOS_PREFIX, headers=headers)
    assert res.status_code == status_code
    assert res.json() == res_body


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, data, status_code, res_body', [
    (None, {}, 401, {'detail': 'Unauthorized'}),
    (
        lazy_fixture('user_token_headers'),
        {'content': 'Play Smash Bros', 'priority_id': 1, 'categories_ids': [1, 4]},
        400,
        {'detail': 'categories are not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        {'content': 'Play Smash Bros', 'priority_id': 1, 'categories_ids': [1, 1]},
        400,
        {'detail': 'categories are not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        {'content': 'Play Smash Bros', 'priority_id': 1, 'categories_ids': [1, 8]},
        400,
        {'detail': 'categories are not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        {'content': 'Play Smash Bros', 'priority_id': 4, 'categories_ids': [1, 2]},
        400,
        {'detail': 'priority is not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        {'content': 'Play Smash Bros', 'priority_id': 1, 'categories_ids': [1, 3]},
        201,
        {
            # id is 4 and not 3 because the 'authorized access non existing priority'
            # test case promotes the primary key.
            'id': 4,
            'is_completed': False,
            'content': 'Play Smash Bros',
            'priority': {'id': 1, 'name': 'Low'},
            'categories': [
                {
                    'id': 1,
                    'name': 'Personal'
                },
                {
                    'id': 3,
                    'name': 'Chess'
                }
            ]
        }
    ),
], ids=[
    'unauthorized access',
    'authorized access another users category',
    'authorized access duplicate valid category',
    'authorized access non existing category',
    'authorized access non existing priority',
    'authorized access valid data'
])
async def test_add_todo(
    client: AsyncClient,
    headers,
    data,
    status_code,
    res_body
):
    res = await client.post(API_TODOS_PREFIX, headers=headers, json=data)
    assert res.status_code == status_code
    assert res.json() == res_body


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, todo_id, data, status_code, res_body', [
    (None, 1, {}, 401, {'detail': 'Unauthorized'}),
    (
        lazy_fixture('user_token_headers'),
        3,
        {'content': 'Learn the sicilian', 'is_completed': True, 'priority_id': 3, 'categories_ids': [2]},
        404,
        {'detail': 'todo does not exist'}
    ),
    (
        lazy_fixture('user_token_headers'),
        2,
        {'content': 'Learn the sicilian', 'is_completed': True, 'priority_id': 3, 'categories_ids': [2]},
        403,
        {'detail': 'a user can not update a todo that was not created by him'}
    ),
    (
        lazy_fixture('user_token_headers'),
        1,
        {'content': 'Learn the sicilian opening', 'is_completed': True, 'priority_id': 2, 'categories_ids': [1, 4]},
        400,
        {'detail': 'categories are not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        1,
        {'content': 'Learn the sicilian opening', 'is_completed': True, 'priority_id': 2, 'categories_ids': [1, 1]},
        400,
        {'detail': 'categories are not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        1,
        {'content': 'Learn the sicilian opening', 'is_completed': True, 'priority_id': 2, 'categories_ids': [1, 8]},
        400,
        {'detail': 'categories are not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        1,
        {'content': 'Learn the sicilian opening', 'is_completed': True, 'priority_id': 5, 'categories_ids': [2]},
        400,
        {'detail': 'priority is not valid'}
    ),
    (
        lazy_fixture('user_token_headers'),
        1,
        {'content': 'Learn the sicilian', 'is_completed': True, 'priority_id': 1, 'categories_ids': [2]},
        200,
        {
            'id': 1,
            'is_completed': True,
            'content': 'Learn the sicilian',
            'priority': {'id': 1, 'name': 'Low'},
            'categories': [
                {
                    'id': 2,
                    'name': 'Work'
                }
            ]
        }
    )

], ids=[
    'unauthorized access',
    'authorized access non existing todo',
    'authorized access another users todo',
    'authorized access another users category',
    'authorized access duplicate valid category',
    'authorized access non existing category',
    'authorized access non existing priority',
    'authorized access valid data'
])
async def test_update_todo(
    client: AsyncClient,
    headers,
    todo_id,
    data,
    status_code,
    res_body
):
    res = await client.put(f'{API_TODOS_PREFIX}/{todo_id}', headers=headers, json=data)
    assert res.status_code == status_code
    assert res.json() == res_body


@pytest.mark.asyncio
@pytest.mark.parametrize('headers, todo_id, status_code, res_body', [
    (None, 1, 401, {'detail': 'Unauthorized'}),
    (lazy_fixture('user_token_headers'), 5, 404, {'detail': 'todo does not exist'}),
    (
        lazy_fixture('user_token_headers'),
        2,
        403,
        {'detail': 'a user can not delete a todo that was not created by him'}
    )
], ids=[
    'unauthorized access',
    'authorized access non existing todo',
    'authorized access another users todo'
])
async def test_delete_todo_failure(
    client: AsyncClient,
    headers,
    todo_id,
    status_code,
    res_body
):
    res = await client.delete(f'{API_TODOS_PREFIX}/{todo_id}', headers=headers)
    assert res.status_code == status_code
    assert res.json() == res_body


@pytest.mark.asyncio
async def test_delete_todo_success(
    client: AsyncClient,
    user_token_headers: dict[str, str]
):
    res = await client.delete(f'{API_TODOS_PREFIX}/1', headers=user_token_headers)
    assert res.status_code == 204
    assert len(res.content) == 0



# ---
# File: /todos/tests/__init__.py
# ---




# ---
# File: /todos/tests/conftest_utils.py
# ---

from typing import Final, Union
import contextlib
import json

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.users.users import get_user_db, get_user_manager
from app.schemas import UserCreate
from app.models.tables import Priority, Category, Todo, TodoCategory, User
from app.core.config import get_config


TESTS_DATA_FILE_PATH: Final[str] = 'todos/tests/tests_data.json'

get_user_db_context = contextlib.asynccontextmanager(get_user_db)
get_user_manager_context = contextlib.asynccontextmanager(get_user_manager)


def get_tests_data() -> dict[str, Union[list[dict], dict]]:
    with open(TESTS_DATA_FILE_PATH, 'r') as f:
        tests_data: dict[str, Union[list[dict], dict]] = json.load(f)
        return tests_data


async def insert_test_data(session: AsyncSession) -> None:
    tests_data = get_tests_data()
    priorities: list[Priority] = [
        Priority(name=p['name']) for p in tests_data['priorities']
    ]
    categories: list[Category] = [
        Category(name=c['name'], created_by_id=None) for c in tests_data['categories']
    ]
    session.add_all(priorities)
    session.add_all(categories)
    for user in tests_data['users']:
        async with get_user_db_context(session) as user_db:
            async with get_user_manager_context(user_db) as user_manager:
                db_user: User = await user_manager.create(
                    UserCreate(email=user['email'], password=user['password'])
                )
        users_categories: list[Category] = [
            Category(name=c['name'], created_by_id=db_user.id) for c in user['categories']
        ]
        session.add_all(users_categories)
        todos: list[Todo] = [
            Todo(
                content=t['content'],
                priority_id=t['priority']['id'],
                created_by_id=db_user.id,
                todos_categories=[TodoCategory(category_id=c['id']) for c in t['categories']]
            ) for t in user['todos']
        ]
        session.add_all(todos)
    await session.commit()


config = get_config()


async def get_user_token_headers(client: AsyncClient) -> dict[str, str]:
    tests_data = get_tests_data()
    login_data = {
        'username': tests_data['users'][0]['email'],
        'password': tests_data['users'][0]['password'],
    }
    res = await client.post(f'{config.API_V1_STR}/auth/login', data=login_data)
    access_token = res.json()['access_token']
    return {'Authorization': f'Bearer {access_token}'}



# ---
# File: /todos/tests/conftest.py
# ---

from typing import Final

import asyncio
from httpx import AsyncClient
from asgi_lifespan import LifespanManager
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.core.db import engine, get_async_session
from app.main import app
from tests.conftest_utils import insert_test_data, get_user_token_headers


@pytest_asyncio.fixture(scope='session', autouse=True)
async def create_test_data():
    async with engine.begin() as conn:
        async with AsyncSession(conn, expire_on_commit=False) as async_session_:
            await insert_test_data(async_session_)


@pytest_asyncio.fixture(scope='session', autouse=True)
def event_loop():
    event_loop_policy = asyncio.get_event_loop_policy()
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture()
async def connection():
    async with engine.begin() as conn:
        yield conn
        await conn.rollback()


@pytest_asyncio.fixture()
async def async_session(connection: AsyncConnection):
    async with AsyncSession(connection, expire_on_commit=False) as async_session_:
        yield async_session_


@pytest_asyncio.fixture(autouse=True)
async def override_dependency(async_session: AsyncSession):
    app.dependency_overrides[get_async_session] = lambda: async_session


TEST_BASE_URL: Final[str] = 'http://test'


@pytest_asyncio.fixture()
async def client():
    async with AsyncClient(app=app, base_url=TEST_BASE_URL) as ac, LifespanManager(app):
        yield ac


@pytest_asyncio.fixture()
async def user_token_headers(client: AsyncClient) -> dict[str, str]:
    return await get_user_token_headers(client)



