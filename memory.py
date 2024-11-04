import torch

def show_memory():
  # 현재 메모리 상태를 보여주는 코드
  gpu_stats = torch.cuda.get_device_properties(0)  # i번째 GPU 속성 가져오기
  start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)  # 현재 예약된 GPU 메모리 계산
  print(f"{start_gpu_memory} GB of memory reserved.")  # 예약된 메모리 양 출력

  max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)  # GPU의 최대 메모리 계
  print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")  # GPU 이름과 최대 메모리 출력

def print_trainable_parameters(bits, model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  if bits == 4: trainable_params /= 2
  print(
    f"trainable params: {trainable_params} || "
    f"all params: {all_param} || "
    f"trainable: {100 * trainable_params / all_param}%"
  )
