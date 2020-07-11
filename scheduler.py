periods = 8
min_class_size = 10
max_class_size = 20
pg = False

def import_pygame():
    pass


class Student:
    def __init__(self, class_requests):
        self.free_periods = list(range(periods))
        self.filled_periods = []
        self.class_requests = class_requests

    def draw_schedule(self):
        pass


class Scheduler:
    def __init__(self, students):
        self.students = students
        self.class_counts = {}
        self.class_to_student = {}
        self.num_class = {}
        self.fill_data()
        self.solve()

    def fill_data(self):
        for student in self.students:
            for class_request in student.class_requests:
                if class_request in self.class_counts:
                    self.class_counts[class_request] += 1
                else:
                    self.class_counts[class_request] = 1
                if class_request in self.class_to_student:
                    self.class_counts[class_request].append(student)
                else:
                    self.class_counts[class_request] = [student]
        for _class in self.class_counts:
            count = self.class_counts[_class]
            self.num_class[_class] = (count//min_class_size, count//max_class_size) \
                if count > min_class_size else (count, count)

    def solve(self):
        for _class in self.class_to_student:
            min_classs_count, max_class_count = None

    def solve_recursive(self, students, unfilled_classes):
        for _class in unfilled_classes:
            pass
