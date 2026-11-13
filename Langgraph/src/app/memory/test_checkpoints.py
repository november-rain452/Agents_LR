from inmemorysaver import history

with open("logs.txt", "a", encoding="utf-8") as file:
    for snapshot in history:
        print("State:", snapshot.values)
        file.write("State:" + str(snapshot.values))
        print("Next:", snapshot.next)
        file.write("Next:" + str(snapshot.next))
        print("----------------")
        file.write("----------------\n")
