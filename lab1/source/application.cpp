#include "application.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif

namespace application {

namespace {

using graphics::internal::context;



constexpr uint32_t kSegments = 32;    
constexpr float kRadius = 1.0f;
constexpr float kHeight = 2.0f;
constexpr uint32_t kObjectCount = 2; 

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

struct UniformData {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec4 tint;
};

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
};

struct SceneObject {
	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 rotation_deg = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);
	float color[3] = { 1.0f, 1.0f, 1.0f };

	Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
	UniformData uniform = {};
};

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
Buffer vertex_buffer;
Buffer index_buffer;
uint32_t index_count = 0;

std::array<SceneObject, kObjectCount> objects;

bool perspective = true;  // доп. 1: тип проекции

bool anim_playing = true;
float anim_speed = 1.0f;
float anim_time = 0.0f;
float traj_radius = 1.0f;
float traj_height = 1.0f;
double last_time = -1.0;


bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo info = {};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer\n";
		return false;
	}

	out.mapped = info.pMappedData;
	return true;
}

void destroyBuffer(Buffer& b) {
	if (b.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, b.buffer, b.allocation);
		b = Buffer{};
	}
}

void uploadToBuffer(const Buffer& b, const void* data, size_t size) {
	std::memcpy(b.mapped, data, size);
	vmaFlushAllocation(context.allocator, b.allocation, 0, VK_WHOLE_SIZE);
}

void buildConeMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	const float half_height = kHeight * 0.5f;

	auto colorFor = [](const glm::vec3& p) {
		return glm::vec3(p.x / kRadius * 0.5f + 0.5f,
		                 p.y / kHeight + 0.5f,
		                 p.z / kRadius * 0.5f + 0.5f);
	};

	for (uint32_t i = 0; i < kSegments; ++i) {
		const float angle = glm::two_pi<float>() * float(i) / float(kSegments);
		const glm::vec3 p(kRadius * std::cos(angle), -half_height, kRadius * std::sin(angle));
		vertices.push_back({ p, colorFor(p) });
	}

	const uint32_t apex = uint32_t(vertices.size());
	const glm::vec3 apex_pos(0.0f, half_height, 0.0f);
	vertices.push_back({ apex_pos, colorFor(apex_pos) });

	const uint32_t center = uint32_t(vertices.size());
	const glm::vec3 center_pos(0.0f, -half_height, 0.0f);
	vertices.push_back({ center_pos, colorFor(center_pos) });

	for (uint32_t i = 0; i < kSegments; ++i) {
		const uint32_t next = (i + 1) % kSegments;
		indices.insert(indices.end(), { apex, next, i });    
		indices.insert(indices.end(), { center, i, next });  
	}
}

VkShaderModule loadShaderModule(const char* file_name) {
	const std::string path = std::string(SHADER_DIR) + file_name;

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader " << path << " (is it compiled by glslc?)\n";
		return VK_NULL_HANDLE;
	}

	const size_t size = size_t(file.tellg());
	std::vector<uint32_t> code(size / 4);
	file.seekg(0, std::ios::beg);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module " << file_name << '\n';
	}
	return module;
}



bool createMesh() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	buildConeMesh(vertices, indices);
	index_count = uint32_t(indices.size());

	const size_t vertices_size = sizeof(Vertex) * vertices.size();
	const size_t indices_size = sizeof(uint32_t) * indices.size();

	if (!createBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
		return false;
	}

	uploadToBuffer(vertex_buffer, vertices.data(), vertices_size);
	uploadToBuffer(index_buffer, indices.data(), indices_size);
	return true;
}


bool createDescriptors() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = kObjectCount,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = kObjectCount,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	std::array<VkDescriptorSetLayout, kObjectCount> layouts;
	layouts.fill(descriptor_set_layout);

	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = kObjectCount,
		.pSetLayouts = layouts.data(),
	};

	std::array<VkDescriptorSet, kObjectCount> sets;
	if (vkAllocateDescriptorSets(context.device, &allocate_info, sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	std::array<VkDescriptorBufferInfo, kObjectCount> buffer_infos;
	std::array<VkWriteDescriptorSet, kObjectCount> writes;

	for (uint32_t i = 0; i < kObjectCount; ++i) {
		if (!createBuffer(sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  objects[i].uniform_buffer)) {
			return false;
		}
		objects[i].descriptor_set = sets[i];

		buffer_infos[i] = VkDescriptorBufferInfo{
			objects[i].uniform_buffer.buffer, 0, sizeof(UniformData)
		};

		writes[i] = VkWriteDescriptorSet{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[i],
		};
	}

	vkUpdateDescriptorSets(context.device, kObjectCount, writes.data(), 0, nullptr);
	return true;
}

bool createPipeline() {
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	VkShaderModule vert = loadShaderModule("cone.vert.spv");
	VkShaderModule frag = loadShaderModule("cone.frag.spv");
	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vert, nullptr);
		vkDestroyShaderModule(context.device, frag, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(Vertex, position)) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(Vertex, color)) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};


	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &pipeline);


	vkDestroyShaderModule(context.device, vert, nullptr);
	vkDestroyShaderModule(context.device, frag, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}
	return true;
}


void drawInterface() {
	ImGui::Begin("Cone");


	int type = perspective ? 0 : 1;
	ImGui::RadioButton("Perspective", &type, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &type, 1);
	perspective = (type == 0);

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(anim_playing ? "Pause" : "Play")) {
		anim_playing = !anim_playing;
	}
	ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Trajectory radius", &traj_radius, 0.0f, 3.0f);
	ImGui::SliderFloat("Vertical amplitude", &traj_height, 0.0f, 3.0f);


	for (uint32_t i = 0; i < kObjectCount; ++i) {
		SceneObject& obj = objects[i];
		const char* names[kObjectCount] = { "Cone 1", "Cone 2" };

		ImGui::PushID(int(i));
		ImGui::SeparatorText(names[i]);
		ImGui::DragFloat3("Position", &obj.position.x, 0.05f, -10.0f, 10.0f);
		ImGui::SliderFloat3("Rotation (deg)", &obj.rotation_deg.x, -180.0f, 180.0f);
		ImGui::SliderFloat3("Scale", &obj.scale.x, 0.1f, 3.0f);
		ImGui::ColorEdit3("Color", obj.color);
		ImGui::PopID();
	}

	ImGui::End();
}

void updateUniforms() {
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 2.0f, 7.0f),
	                                   glm::vec3(0.0f),
	                                   glm::vec3(0.0f, 1.0f, 0.0f));


	glm::mat4 projection(1.0f);
	if (perspective) {
		projection = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
	} else {
		const float half_height = 4.0f;
		const float half_width = half_height * aspect;
		projection = glm::ortho(-half_width, half_width, -half_height, half_height, 0.1f, 100.0f);
	}
	projection[1][1] *= -1.0f;

	const glm::mat4 identity(1.0f);
	const glm::vec3 spin_axis = glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f));

	for (SceneObject& obj : objects) {
		const float t = anim_time;
		const glm::vec3 position = obj.position + glm::vec3(traj_radius * std::sin(t),
		                                                    traj_height * std::sin(2.0f * t),
		                                                    traj_radius * std::cos(t));
		const glm::mat4 animation_rotation = glm::rotate(identity, 1.5f * t, spin_axis);

		const glm::mat4 rotation =
			glm::rotate(identity, glm::radians(obj.rotation_deg.z), glm::vec3(0.0f, 0.0f, 1.0f)) *
			glm::rotate(identity, glm::radians(obj.rotation_deg.y), glm::vec3(0.0f, 1.0f, 0.0f)) *
			glm::rotate(identity, glm::radians(obj.rotation_deg.x), glm::vec3(1.0f, 0.0f, 0.0f));

		obj.uniform.model = glm::translate(identity, position) * animation_rotation * rotation *
		                    glm::scale(identity, obj.scale);
		obj.uniform.view = view;
		obj.uniform.projection = projection;
		obj.uniform.tint = glm::vec4(obj.color[0], obj.color[1], obj.color[2], 1.0f);
	}
}

} 

bool initialize() {
	objects[0].position = glm::vec3(-2.0f, 0.0f, 0.0f);
	objects[1].position = glm::vec3(2.0f, 0.0f, 0.0f);
	objects[1].color[0] = 1.0f;
	objects[1].color[1] = 0.6f;
	objects[1].color[2] = 0.3f;

	if (!createMesh() || !createDescriptors() || !createPipeline()) {
		shutdown();  
		return false;
	}
	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);  
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	for (SceneObject& obj : objects) {
		destroyBuffer(obj.uniform_buffer);
	}
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);
}

void update(double time) {
	if (last_time < 0.0) {
		last_time = time;
	}
	const float dt = float(time - last_time);
	last_time = time;

	if (anim_playing) {
		anim_time += dt * anim_speed;
	}

	drawInterface();
	updateUniforms();
}

void render(const graphics::internal::FrameData& fd) {
	VkCommandBuffer cmd = fd.command_buffer;

	for (SceneObject& obj : objects) {
		uploadToBuffer(obj.uniform_buffer, &obj.uniform, sizeof(UniformData));
	}

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2] = {};
	clear_values[0].color = { { 0.08f, 0.09f, 0.11f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	const VkRect2D scissor = { .offset = { 0, 0 }, .extent = context.swapchain_extent };
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	for (SceneObject& obj : objects) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
		                        &obj.descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application