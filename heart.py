# Code by Quyet Thanh - PTIT student
# Facebook: https://www.facebook.com/quyethanh.dvfb
# Follow me <3

# Khai báo thư viện Turtle:
import turtle

# Tạo đối tượng con trỏ (hoặc bút) để vẽ:
cursor = turtle.Turtle()

# Đặt màu cho con trỏ:
cursor.fillcolor('pink')
cursor.begin_fill()

# Hàm vẽ đường cong:
def uon_cong(n):
	for i in range(n):
		cursor.right(1)
		cursor.forward(1)

# Viết chữ:
def put_txt(text):
	# Nhấc bút lên để không bị lem mực:
	cursor.up()
	# Đặt bút tại vị tr giữa trái tim:
	cursor.setpos(-58, 95)
	cursor.down()

	# Đặt màu chữ và viết chữ:
	cursor.color('white')
	cursor.write(text, font= ("Vedana", 12, "bold"))
	# Ẩn con trỏ vẽ:
	cursor.hideturtle()

def nhip_tim():
	cursor.up()
	cursor.left(180)
	cursor.forward(380)
	cursor.right(180)
	cursor.down()
	cursor.forward(135)
	cursor.left(60)
	cursor.forward(10)
	cursor.right(120)
	cursor.forward(15)
	cursor.left(60)
	cursor.forward(5)
	cursor.right(30)
	cursor.forward(6)
	cursor.left(60)
	cursor.forward(8)
	cursor.right(30)
	cursor.forward(3)
	cursor.left(80)
	cursor.forward(60)
	cursor.right(160)
	cursor.forward(162)
	cursor.left(160)
	cursor.forward(105)
	cursor.right(80)
	cursor.forward(150)
	ve_tym()
	cursor.left(140)
	cursor.forward(150)
	cursor.right(80)
	cursor.forward(105)
	cursor.left(160)
	cursor.forward(162)
	cursor.right(160)
	cursor.forward(60)
	cursor.left(80)
	cursor.forward(3)
	cursor.right(30)
	cursor.forward(8)
	cursor.left(60)
	cursor.forward(6)
	cursor.right(30)
	cursor.forward(5)
	cursor.left(60)
	cursor.forward(15)
	cursor.right(120)
	cursor.forward(10)
	cursor.left(60)
	cursor.forward(135)
	cursor.hideturtle()
def ve_tym():
	cursor.fillcolor('red')
	cursor.begin_fill()

	cursor.left(140)
	cursor.forward(113)
	# Vẽ đường cong trên:
	uon_cong(200)

	# Thay đổi hướng vẽ:
	cursor.left(120)

	# Vẽ tiếp đường cong bên phải:
	uon_cong(200)

	cursor.forward(110)
	#Tô màu cho trái tim:
	cursor.fillcolor('red')
	cursor.end_fill()

#Vẽ nhịp tim và trái tim
nhip_tim()
#Viết chữ và trong trái tim
put_txt("I love you!")
# Nếu nhấn vào cửa sổ thì kết thúc chương trình:
turtle.exitonclick()
