from commons import *

def main():
    start_station = "KRZYKI"
    end_station = "Chłodna"
    start_time = datetime.strptime("8:00:00 PM", TIME_FORMAT)

    graph = load_graph()
    print("Graph loaded")
    
    
    print("Dijkstra with t parameter: ")
    ans = graph.dijkstra(start_station, end_station, 't', start_time)
    print(ans)
if __name__ == '__main__':
    main()