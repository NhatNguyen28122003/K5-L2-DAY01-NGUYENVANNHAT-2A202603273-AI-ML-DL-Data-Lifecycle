# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4GPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
Record này nói về cả bức ảnh không phải nói về một vật thể riêng trong ảnh  
Vì record này mô tả kết quả phân loại toàn bộ bức ảnh (Image Classification), không phải một vật thể riêng biệt. Vì trường task có giá trị image_classification, cho biết mô hình thực hiện nhiệm vụ phân loại ảnh. Kết quả class_name = cab thể hiện lớp được mô hình dự đoán cho ảnh đầu vào
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
Danh sách lớp không phải do mô hình tự nghĩ ra mà được xác định trước trong quá trình xây dựng và huấn luyện mô hình. Taxonomy quy định tập hợp các lớp hợp lệ, ví dụ ImageNet-1K gồm 1.000 lớp. Checkpoint lưu các trọng số đã huấn luyện và thường chứa hoặc liên kết với thông tin tên lớp tương ứng. Khi dự đoán, mô hình tính điểm cho các lớp đã được định nghĩa và chọn lớp phù hợp nhất.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
Cần lưu cả ba trường vì mỗi trường có chức năng riêng. class_id là mã định danh giúp máy tính lưu trữ và xử lý dữ liệu; class_name là tên lớp giúp con người dễ hiểu kết quả; taxonomy_name xác định hệ thống phân loại mà ID và tên lớp thuộc về. Việc kết hợp ba trường giúp tránh nhầm lẫn khi có nhiều taxonomy sử dụng các ID giống nhau hoặc có cách định nghĩa lớp khác nhau, đồng thời hỗ trợ quản lý, đối chiếu và tái sử dụng dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
Guideline cần quy định rõ tiêu chí lựa chọn lớp đại diện cho toàn bộ bức ảnh, chẳng hạn xác định chủ thể chính dựa trên mục tiêu của bộ dữ liệu, mức độ nổi bật về ngữ nghĩa hoặc vai trò của đối tượng. Đồng thời, cần quy định cách xử lý khi ảnh có nhiều đối tượng quan trọng tương đương, ảnh mơ hồ hoặc không thuộc danh sách lớp hợp lệ. Ngoài ra, cần có quy trình kiểm duyệt khi các annotator bất đồng. Điều này giúp đảm bảo tính nhất quán của dữ liệu gán nhãn và hạn chế sai lệch trong quá trình huấn luyện mô hình.
6. Vì sao không được lấy `score` của model làm ground truth?  
Không được sử dụng score của mô hình làm Ground Truth vì score chỉ thể hiện mức độ tự tin hoặc điểm số mà mô hình gán cho một lớp dự đoán, không chứng minh dự đoán đó chính xác. Mô hình vẫn có thể đưa ra dự đoán sai với score rất cao do dữ liệu thiếu đa dạng, ảnh không rõ hoặc đặc điểm giữa các lớp tương tự nhau. Ground Truth phải được xác định độc lập thông qua quá trình gán nhãn theo guideline và kiểm duyệt phù hợp. Đây là căn cứ để đánh giá mô hình bằng các chỉ số như Accuracy, Precision và Recall.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person; `score`= 0.912625; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
Bounding box bao quanh người được phát hiện trong ảnh:

- Góc trên-trái (x1, y1): (385.33, 69.24) pixel.

- Góc dưới-phải (x2, y2): (498.92, 348.92) pixel.

- Chiều rộng: 113.58 pixel.

- Chiều cao: 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  

- Ngưỡng 0.20 có 11

- Ngưỡng 0.60 có 6


4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
Khi hạ confidence threshold, model có thể giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán không chính xác.

Reviewer cần:

- Kiểm tra thêm các bounding box mới xuất hiện.

- Loại bỏ các box nhận diện sai đối tượng (false positives).

- Kiểm tra các box bị trùng lặp.

- Xác nhận class và vị trí box.

- Đảm bảo không bỏ sót đối tượng thực tế.

Như vậy, ngưỡng càng thấp thì reviewer thường càng phải kiểm tra nhiều hơn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
Quy tắc: Vẽ bounding box nhỏ nhất có thể nhưng phải bao phủ toàn bộ phần đối tượng nhìn thấy trong ảnh. Box phải sát các mép ngoài của đối tượng, không bao gồm quá nhiều nền và không được cắt mất phần đối tượng đang nhìn thấy.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
Đưa vào guideline nếu đây là trường hợp phổ biến và có thể mô tả rõ ràng.

Ví dụ: Một vật dụng nhà bếp nằm sát mép ảnh và chỉ xuất hiện một phần.

Quy tắc đề xuất:

- Nếu vật bị cắt bởi mép ảnh, chỉ vẽ bounding box bao quanh phần thực sự nhìn thấy.

- Không tự suy đoán hoặc vẽ thêm phần nằm ngoài khung hình.

- Nếu không thể xác định chắc chắn vật thể thuộc class nào, cần đánh dấu để reviewer kiểm tra.

Vì:

- Một quy tắc rõ ràng giúp người gán nhãn xử lý nhất quán các trường hợp vật thể bị cắt mép, giảm sự khác biệt giữa các annotator.

- Nếu xuất hiện trường hợp bất thường, không có hướng dẫn rõ ràng hoặc ảnh hưởng đến cách phân loại, nên escalation lên Lab Coach để thống nhất trước khi áp dụng rộng rãi.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735744 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[
    [53.0, 344.0],
    [52.0, 345.0],
    [50.0, 345.0],
    [49.0, 346.0],
    [48.0, 346.0]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
Bounding box chỉ xác định vùng hình chữ nhật bao quanh vật thể, trong khi polygon mô tả đường biên và hình dạng chi tiết của vật thể. Polygon giúp phân biệt chính xác vùng thuộc vật thể với vùng nền, đặc biệt đối với các vật thể có hình dạng không đều.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
instance_id là mã định danh giúp phân biệt từng cá thể đối tượng trong dữ liệu, kể cả khi chúng có cùng class. Nó khác class_id, vốn xác định loại đối tượng, và khác tracking_id, vốn phục vụ theo dõi cùng một đối tượng qua nhiều frame video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
Vẽ polygon theo đường viền của phần vật thể nhìn thấy được, đặt các điểm sát biên ngoài của vật thể, không bao gồm vùng nền và không tự suy đoán phần bị che khuất. Khi biên không rõ ràng hoặc không thể xác định vật thể thuộc về đâu, thực hiện theo guideline đã được phê duyệt; nếu guideline chưa quy định trường hợp đó, chuyển escalation để được hướng dẫn.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
Trong ảnh kitchen, tôi chọn chiếc bát sứ màu trắng ở góc dưới bên trái. Phần biên phía dưới và bên phải cần chú ý vì màu sắc của bát tương đối gần với mặt bàn. Tôi sẽ áp dụng guideline để vẽ polygon theo đường viền thực tế của phần bát nhìn thấy được, không đưa nền vào mask. Nếu guideline chưa quy định cách xử lý hoặc không thể xác định chính xác đường biên, tôi sẽ escalation cho Lab Coach để được hướng dẫn, nhằm đảm bảo tính nhất quán của dữ liệu gán nhãn.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  Một hoặc nhiều nhãn cấp ảnh, ví dụ: kitchen, kitchen interior. Đơn vị nhãn là toàn bộ ảnh.|JSON dự đoán class_name = cab, score = 0.510915. Nhưng PNG thể hiện một căn bếp, không phải xe taxi. Đây là dấu hiệu prediction có thể phân loại sai hoặc taxonomy/nhãn chưa phù hợp.  |Xác định nội dung chính của ảnh và chọn nhãn đúng theo taxonomy, không chọn theo prediction một cách máy móc.  |Kiểm tra nhãn có phản ánh nội dung toàn ảnh không; kiểm tra taxonomy và các trường hợp ảnh có nhiều chủ thể.  |
| Phát hiện vật thể | Danh sách từng vật thể, mỗi nhãn gồm class + bounding box. Ví dụ: person + tọa độ xyxy. |JSON dự đoán person, score = 0.912625, bbox_xyxy = [385.33, 69.24, 498.92, 348.92]. Người thực sự xuất hiện ở vùng giữa bên phải ảnh. Cần kiểm tra box có ôm sát phần người nhìn thấy không, có thừa nền hoặc thiếu bộ phận không.  |Xác định từng vật thể thuộc danh sách lớp; vẽ bounding box quanh phạm vi vật thể nhìn thấy; không bỏ sót hoặc gộp các vật thể riêng biệt.  |Kiểm tra class, vị trí và kích thước box; phát hiện box sai, thiếu vật thể, box trùng lặp hoặc false positive.  |
| Instance segmentation |  Danh sách từng cá thể, mỗi nhãn gồm class + mask/polygon. Ví dụ: bowl + polygon mô tả vùng chiếc bát.|JSON dự đoán class_name = bowl, instance_id = kitchen-002, polygon_point_count = 67. PNG có chiếc bát sứ trắng ở góc dưới bên trái; biên phía dưới và bên phải tương đối khó phân biệt với mặt bàn. Cần kiểm tra polygon có lấy nhầm nền không.  |Vẽ mask theo đường biên phần vật thể nhìn thấy; tách riêng từng cá thể cùng lớp; tuân thủ quy tắc xử lý vật thể bị che hoặc có biên mờ.  |Kiểm tra class, mask có bám sát biên không, có bỏ sót pixel hoặc lấy nhầm nền không; kiểm tra việc tách các instance.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
Chỉ sử dụng dữ liệu trong phạm vi được cấp phép cho bài lab, không tự ý sao chép, tải xuống, chia sẻ hoặc tiết lộ dữ liệu cho người không có thẩm quyền.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
Dừng xử lý, không tự ý sử dụng hoặc chia sẻ dữ liệu đó, đồng thời báo ngay cho người hướng dẫn (Lab Coach) hoặc người phụ trách dự án để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
