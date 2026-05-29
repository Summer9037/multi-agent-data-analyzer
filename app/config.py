"""集中配置：从 .env 加载所有运行时参数。

所有模块统一从这里 `from app.config import settings` 取值，避免到处读环境变量。
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(None, alias="ANTHROPIC_BASE_URL")
    anthropic_model_planner: str = Field("claude-opus-4-7", alias="ANTHROPIC_MODEL_PLANNER")
    anthropic_model_reporter: str = Field("claude-opus-4-7", alias="ANTHROPIC_MODEL_REPORTER")
    anthropic_model_worker: str = Field("claude-sonnet-4-6", alias="ANTHROPIC_MODEL_WORKER")

    # PostgreSQL
    db_host: str = Field("localhost", alias="DB_HOST")
    db_port: int = Field(5432, alias="DB_PORT")
    db_user: str = Field("postgres", alias="DB_USER")
    db_password: str = Field("postgres", alias="DB_PASSWORD")
    db_name: str = Field("auto_analysis", alias="DB_NAME")

    # 数据路径
    raw_data_dir: Path = Field(..., alias="RAW_DATA_DIR")
    output_dir: Path = Field(Path("./outputs"), alias="OUTPUT_DIR")

    # 数据规模
    raw_sample_size: int = Field(300_000, alias="RAW_SAMPLE_SIZE")

    # Agent
    agent_max_iterations: int = Field(10, alias="AGENT_MAX_ITERATIONS")

    @property
    def db_url(self) -> str:
        # psycopg3 dialect: postgresql+psycopg://
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def db_connect_args(self) -> dict:
        """传给 psycopg.connect / SQLAlchemy create_engine 的 connect_args。

        psycopg3 把 client_encoding 写进 startup packet，服务端从握手开始
        就用 UTF-8 编码所有 ParameterStatus 与消息，从根上避开中文 Windows PG
        的 GBK 解码问题。options 同时强制服务端 lc_messages=C（英文消息）。
        """
        return {
            "client_encoding": "UTF8",
            "options": "-c lc_messages=C",
        }


settings = Settings()  # type: ignore[call-arg]
