# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


@dataclass
class ToolCallCompressionSegment:
    """Token span for one tool-processing round in the post-initial-prompt trajectory."""

    start_offset: int
    end_offset: int
    tool_call_count: int
    image_count: int = 0


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.initial_prompt_length = 0
        if image_data is None:
            self.initial_image_count = 0
        elif isinstance(image_data, list):
            self.initial_image_count = len(image_data)
        else:
            self.initial_image_count = 1
        self.tool_call_count: int = 0 # TBR

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.tool_call_segments: list[ToolCallCompressionSegment] = []
        self.history_trajectories: list[AgentLoopOutput] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        agent_data = await self._create_agent_data(**kwargs)

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        return self._build_agent_loop_output(agent_data)

    async def _create_agent_data(self, **kwargs) -> AgentData:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        return AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

    def _copy_multi_modal_data(self, agent_data: AgentData) -> dict[str, Any]:
        multi_modal_data: dict[str, Any] = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = (
                list(agent_data.image_data) if isinstance(agent_data.image_data, list) else agent_data.image_data
            )
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = (
                list(agent_data.video_data) if isinstance(agent_data.video_data, list) else agent_data.video_data
            )
        return multi_modal_data

    def _build_output_extra_fields(self, agent_data: AgentData) -> dict[str, Any]:
        return {
            "turn_scores": list(agent_data.turn_scores),
            "tool_rewards": list(agent_data.tool_rewards),
            "tool_call_count": agent_data.tool_call_count, # TBR
        }

    def _build_agent_loop_output(self, agent_data: AgentData) -> AgentLoopOutput:
        prompt_ids = list(agent_data.prompt_ids[: agent_data.initial_prompt_length])
        response_ids = list(agent_data.prompt_ids[agent_data.initial_prompt_length :])
        response_mask = list(agent_data.response_mask)
        response_logprobs = list(agent_data.response_logprobs) if agent_data.response_logprobs else None

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=self._copy_multi_modal_data(agent_data),
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=dict(agent_data.metrics),
            routed_experts=agent_data.routed_experts,
            extra_fields={},
        )
        output.extra_fields.update(self._build_output_extra_fields(agent_data))
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        agent_data.initial_prompt_length = len(prompt_ids)
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        agent_data.tool_call_count += len(tool_call_names) # TBR

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map


@register("tool_agent_sliding_window_compression")
class ToolAgentLoopWithSlidingWindowCompression(ToolAgentLoop):
    """Tool agent loop that periodically compresses old tool-call rounds in the response trajectory.

    This loop returns multiple `AgentLoopOutput` snapshots and is intended for trainers that already support
    `list[AgentLoopOutput]`, such as the TransferQueue path in `main_ppo_sync.py`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        multi_turn_config = self.rollout_config.multi_turn
        self.compression_trigger_tool_calls = int(multi_turn_config.get("compression_trigger_tool_calls", 10))
        self.compression_remove_tool_calls = int(multi_turn_config.get("compression_remove_tool_calls", 7))
        self.compression_placeholder_text = multi_turn_config.get(
            "compression_placeholder_text", "[Compressed_ToolCall_Tokens]"
        )
        if self.compression_trigger_tool_calls <= 0:
            raise ValueError("compression_trigger_tool_calls must be positive.")
        if self.compression_remove_tool_calls <= 0:
            raise ValueError("compression_remove_tool_calls must be positive.")
        self._compression_placeholder_ids: Optional[list[int]] = None

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        agent_data = await self._create_agent_data(**kwargs)

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        final_output = self._build_agent_loop_output(agent_data)
        final_output.extra_fields["is_compression_snapshot"] = False
        outputs = list(agent_data.history_trajectories)
        outputs.append(final_output)
        return outputs

    def _build_output_extra_fields(self, agent_data: AgentData) -> dict[str, Any]:
        extra_fields = super()._build_output_extra_fields(agent_data)
        extra_fields["history_trajectory_count"] = len(agent_data.history_trajectories)
        return extra_fields

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination=True)
        if state == AgentState.PROCESSING_TOOLS:
            return state
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        assistant_response_len = len(agent_data.response_ids)
        assistant_span_start = len(agent_data.response_mask) - assistant_response_len

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        agent_data.tool_call_count += len(tool_call_names)

        for tool_response, tool_reward, _ in responses:
            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_response.image:
                if isinstance(tool_response.image, list):
                    for img in tool_response.image:
                        if img is not None:
                            new_images_this_turn.append(img)
                elif tool_response.image is not None:
                    new_images_this_turn.append(tool_response.image)

            if tool_response.video:
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        response_ids = await self._build_tool_response_ids(add_messages, tool_call_names, new_images_this_turn)

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        agent_data.tool_call_segments.append(
            ToolCallCompressionSegment(
                start_offset=assistant_span_start,
                end_offset=len(agent_data.response_mask),
                tool_call_count=len(tool_call_names),
                image_count=len(new_images_this_turn),
            )
        )

        while self._should_compress(agent_data):
            self._append_history_trajectory(agent_data)
            await self._compress_old_tool_call_context(agent_data)

        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _build_tool_response_ids(
        self, add_messages: list[dict[str, Any]], tool_call_names: list[str], new_images_this_turn: list[Any]
    ) -> list[int]:
        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            return await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        return await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn if new_images_this_turn else None,
            videos=None,
            remove_system_prompt=True,
        )

    def _append_history_trajectory(self, agent_data: AgentData) -> None:
        snapshot = self._build_agent_loop_output(agent_data)
        snapshot.extra_fields["is_compression_snapshot"] = True
        snapshot.extra_fields["history_trajectory_index"] = len(agent_data.history_trajectories)
        agent_data.history_trajectories.append(snapshot)

    def _should_compress(self, agent_data: AgentData) -> bool:
        return self._count_tracked_tool_calls(agent_data) >= self.compression_trigger_tool_calls

    def _count_tracked_tool_calls(self, agent_data: AgentData) -> int:
        return sum(segment.tool_call_count for segment in agent_data.tool_call_segments)

    async def _get_compression_placeholder_ids(self) -> list[int]:
        if self._compression_placeholder_ids is None:
            placeholder_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(self.compression_placeholder_text, add_special_tokens=False)
            )
            if not placeholder_ids:
                raise ValueError("Compression placeholder text produced no tokens.")
            self._compression_placeholder_ids = list(placeholder_ids)
        return list(self._compression_placeholder_ids)

    async def _compress_old_tool_call_context(self, agent_data: AgentData) -> None:
        segments_to_remove: list[ToolCallCompressionSegment] = []
        removed_tool_calls = 0
        for segment in agent_data.tool_call_segments:
            segments_to_remove.append(segment)
            removed_tool_calls += segment.tool_call_count
            if removed_tool_calls >= self.compression_remove_tool_calls:
                break

        if not segments_to_remove:
            return

        start_offset = segments_to_remove[0].start_offset
        end_offset = segments_to_remove[-1].end_offset
        placeholder_ids = await self._get_compression_placeholder_ids()
        placeholder_mask = [0] * len(placeholder_ids)

        response_ids = list(agent_data.prompt_ids[agent_data.initial_prompt_length :])
        agent_data.prompt_ids = (
            list(agent_data.prompt_ids[: agent_data.initial_prompt_length])
            + response_ids[:start_offset]
            + placeholder_ids
            + response_ids[end_offset:]
        )
        agent_data.response_mask = (
            list(agent_data.response_mask[:start_offset])
            + placeholder_mask
            + list(agent_data.response_mask[end_offset:])
        )
        if agent_data.response_logprobs:
            agent_data.response_logprobs = (
                list(agent_data.response_logprobs[:start_offset])
                + [0.0] * len(placeholder_ids)
                + list(agent_data.response_logprobs[end_offset:])
            )

        self._drop_compressed_images(agent_data, sum(segment.image_count for segment in segments_to_remove))

        shift = len(placeholder_ids) - (end_offset - start_offset)
        remaining_segments = []
        for segment in agent_data.tool_call_segments[len(segments_to_remove) :]:
            remaining_segments.append(
                ToolCallCompressionSegment(
                    start_offset=segment.start_offset + shift,
                    end_offset=segment.end_offset + shift,
                    tool_call_count=segment.tool_call_count,
                    image_count=segment.image_count,
                )
            )
        agent_data.tool_call_segments = remaining_segments

    def _drop_compressed_images(self, agent_data: AgentData, removed_image_count: int) -> None:
        if removed_image_count <= 0 or agent_data.image_data is None:
            return

        if not isinstance(agent_data.image_data, list):
            image_data = [agent_data.image_data]
        else:
            image_data = list(agent_data.image_data)

        removable_images = max(len(image_data) - agent_data.initial_image_count, 0)
        removed_image_count = min(removed_image_count, removable_images)
        keep_prefix = image_data[: agent_data.initial_image_count]
        keep_suffix = image_data[agent_data.initial_image_count + removed_image_count :]
        remaining_images = keep_prefix + keep_suffix
        agent_data.image_data = remaining_images if remaining_images else None
