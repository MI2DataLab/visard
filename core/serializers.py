import pandas as pd
from rest_framework import serializers

from .models import Path


class Column(object):
    def __init__(self, colname, data_type, data):
        self.colname = colname
        self.type = data_type
        self.data = data

    def __str__(self):
        return pd.DataFrame({self.colname: self.data}).__str__()


class PathSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Path
        fields = "__all__"


class ColumnSerializer(serializers.Serializer):
    colname = serializers.CharField()
    type = serializers.CharField()
    data = serializers.ListField()

    def create(self, validated_data):
        return Column(**validated_data)

    def update(self, instance, validated_data):
        for field, value in validated_data.items():
            setattr(instance, field, value)
        return instance
