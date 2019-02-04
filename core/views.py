import itertools
import sys

from django.shortcuts import render, get_object_or_404
import pandas as pd
from rest_framework import viewsets, generics

from .serializers import PathSerializer, ColumnSerializer, Column
from .models import Path

thismodule = sys.modules[__name__]
thismodule.DATA = {
    'asd': Column('asd', 'int', [1, 3, 5, 2]),
    'dsf': Column('dsf', 'str', ['a', 'b', 'c']),
}


class ColumnView(generics.ListAPIView):
    serializer_class = ColumnSerializer

    def get_queryset(self):
        """
        Optionally restricts the returned purchases to a given user,
        by filtering against a `colname` query parameter in the URL.
        """
        queryset = thismodule.DATA
        colname = self.request.query_params.get('colname', None)
        if colname is not None:
            queryset = [queryset[colname]]
        else:
            queryset = list(queryset.values())
        return queryset


class PathViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = Path.objects.all().order_by('pk')
    serializer_class = PathSerializer


def upload_data(request):
    if request.method == 'POST':
        data_dir = request.POST.get('data_dir')
        path = Path.objects.get_or_create(path=data_dir)
        return plots(request, path[0].pk)
    else:
        return render(request, 'core/upload_data.html', {})


def data_for_plots(data_dir):
    data = pd.read_csv(data_dir)
    sorted_cols = sort_cols(data)
    sorted_pairs = sort_pairs(data)
    return data, sorted_cols, sorted_pairs


def sort_cols(data):
    return data.columns


def sort_pairs(data):
    return list(itertools.combinations(data.columns, 2))


def data_to_columns(data):
    data_dict = data.to_dict('list')
    data_dict = {name: Column(name, type(val[0]), val) for name, val in data_dict.items()}
    return data_dict


def plots(request, path_id):
    path = get_object_or_404(Path, pk=path_id)
    data, sorted_cols, sorted_pairs = data_for_plots(path.path)
    thismodule.DATA = data_to_columns(data)
    return render(
        request,
        'core/plots.html',
        {'sorted_cols': sorted_cols, 'sorted_pairs': sorted_pairs},
    )
