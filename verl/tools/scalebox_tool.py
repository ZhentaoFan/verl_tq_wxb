# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import time
from typing import Any

import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ScaleBoxTool(BaseTool):
    """Execute code with a ScaleBox-compatible run_code endpoint."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.scalebox_url = config.get("scalebox_url", "")
        if not self.scalebox_url:
            raise ValueError("scalebox_url is not set")
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.max_retries = max(1, int(config.get("max_retries", 4)))
        self.initial_retry_delay = max(0.0, float(config.get("initial_retry_delay", 10.0)))
        retry_status_codes = config.get("retry_status_codes", [429, 500, 502, 503, 504])
        self.retry_status_codes = {int(code) for code in retry_status_codes}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(None, self._execute_sync, code, timeout, language)
        return ToolResponse(text=response_text), None, None

    def _execute_sync(self, code: str, timeout: int, language: str) -> str:
        payload = {"code": code, "language": language}
        last_error = "unknown error"
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.scalebox_url, json=payload, timeout=timeout)
                if response.status_code in self.retry_status_codes:
                    raise requests.HTTPError(f"retryable status code: {response.status_code}", response=response)
                response.raise_for_status()
                data = response.json()
                return self._format_response(data)
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.max_retries - 1 or not self._is_retryable(exc):
                    logger.warning("ScaleBox request failed after %d attempt(s): %s", attempt + 1, exc)
                    return f"ScaleBox error: {exc}"

                delay = self.initial_retry_delay * (2**attempt)
                logger.warning("ScaleBox request failed on attempt %d/%d, retrying in %.2fs: %s", attempt + 1,
                               self.max_retries, delay, exc)
                time.sleep(delay)

        return f"ScaleBox error: {last_error}"

    def _is_retryable(self, exc: Exception) -> bool:
        if not isinstance(exc, requests.RequestException):
            return False
        if exc.response is not None:
            return exc.response.status_code in self.retry_status_codes
        # Connection/timeout errors are retryable by default.
        return True

    def _format_response(self, data: dict[str, Any]) -> str:
        if data.get("status") == "SandboxError":
            return data.get("message", "") or "ScaleBox sandbox error"

        run_result = data.get("run_result") or {}
        stdout = run_result.get("stdout") or ""
        stderr = run_result.get("stderr") or ""
        if stdout or stderr:
            return f"{stdout}{stderr}"

        compile_result = data.get("compile_result") or {}
        compile_stderr = compile_result.get("stderr") or ""
        if compile_stderr:
            return compile_stderr

        return data.get("message", "") or "ScaleBox returned no output"
