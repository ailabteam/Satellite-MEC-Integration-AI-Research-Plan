Bước 1: Mô hình hóa Hệ thống (System Modeling)
Đây là bước biến kịch bản trên thành ngôn ngữ toán học.
Câu hỏi cho bạn: Để mô tả bài toán trên một cách toán học, chúng ta cần định nghĩa những tham số (parameter) và biến (variable) nào?
Hãy thử bắt đầu với những thứ sau, và bạn bổ sung thêm nhé:
Mô tả Tác vụ (Task):
Kích thước dữ liệu cần truyền (ví dụ: D - tính bằng bits).
Lượng tính toán cần thiết (ví dụ: C - tính bằng số chu kỳ CPU).
Mô tả các Nơi xử lý (Computing Nodes):
Drone: Năng lực tính toán f_local (CPU cycles/giây).
Vệ tinh MEC: Năng lực tính toán f_sat (CPU cycles/giây).
Trạm mặt đất MEC: Năng lực tính toán f_ground (CPU cycles/giây).
Mô tả Kết nối mạng (Network Links):
Băng thông từ drone lên vệ tinh: B_up (bits/giây).
Băng thông từ vệ tinh xuống drone: B_down (bits/giây).
Độ trễ truyền dẫn một chiều (propagation delay) từ drone tới vệ tinh: L_prop.
Câu hỏi thảo luận #1: Dựa vào các tham số trên, làm thế nào để chúng ta tính được:
Thời gian xử lý (Computation Time) nếu tác vụ được thực hiện tại drone? Tại vệ tinh?
Thời gian truyền tải (Transmission Time) nếu tác vụ được dỡ tải lên vệ tinh?
Hãy thử viết ra công thức. Đây chính là khởi đầu của việc công thức hóa bài toán.

Bước 2: Công thức hóa Bài toán (Problem Formulation)
Giả sử bạn đã có công thức cho thời gian và năng lượng. Bây giờ chúng ta định nghĩa mục tiêu.
Biến quyết định (Decision Variable): Chúng ta có thể định nghĩa một biến a để thể hiện quyết định. Ví dụ:
a = 0 (chọn xử lý tại chỗ)
a = 1 (chọn dỡ tải lên vệ tinh)
a = 2 (chọn dỡ tải về trạm mặt đất)
Hàm mục tiêu (Objective Function): Chúng ta muốn tối ưu cả thời gian và năng lượng. Một cách phổ biến là dùng một tổng có trọng số:
Minimize: Cost(a) = α * Total_Time(a) + (1-α) * Energy_Consumption(a)
Trong đó α (alpha) là một trọng số từ 0 đến 1, cho thấy chúng ta ưu tiên thời gian hay năng lượng hơn.
Câu hỏi thảo luận #2: Total_Time(a) khi a=1 (dỡ tải lên vệ tinh) sẽ bao gồm những thành phần nào?
Gợi ý: Nó không chỉ có thời gian xử lý.

Bước 3: Tại sao bài toán này "khó" và cần AI?
Nếu tất cả các tham số (B_up, L_prop, f_sat...) đều là hằng số và biết trước, bài toán này rất dễ. Drone chỉ cần tính chi phí Cost cho cả 3 lựa chọn và chọn cái nhỏ nhất.
Nhưng trong thực tế, môi trường lại VÔ CÙNG ĐỘNG (Highly Dynamic):
Vệ tinh di chuyển: Vị trí vệ tinh thay đổi từng giây -> L_prop (độ trễ) và B_up (băng thông, do khoảng cách và góc phương vị thay đổi) cũng thay đổi liên tục.
Tải của Vệ tinh: f_sat không phải lúc nào cũng có sẵn. Server MEC trên vệ tinh có thể đang bận xử lý tác vụ của 10 người dùng khác. Nó có một hàng đợi (queue). Tác vụ của bạn có thể phải chờ.
Kênh truyền không ổn định: Băng thông B_up có thể bị ảnh hưởng bởi thời tiết (mưa), chướng ngại vật...
Đây chính là lúc AI tỏa sáng! Một hệ thống "tĩnh" sẽ không thể đưa ra quyết định tối ưu. Chúng ta cần một "bộ não" có khả năng:
Học từ môi trường đang thay đổi.
Dự đoán trạng thái tương lai (ví dụ: vệ tinh nào sắp bay tới, băng thông có thể sẽ là bao nhiêu).
Ra quyết định tối ưu dựa trên trạng thái hiện tại và dự đoán tương lai.
Đây là một bài toán ra quyết định tuần tự (sequential decision-making) trong môi trường không chắc chắn -> Ứng cử viên hoàn hảo cho Học tăng cường (Reinforcement Learning - RL).

