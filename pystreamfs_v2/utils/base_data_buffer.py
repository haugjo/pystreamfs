class DataBuffer:
    def set_elements(self, **kwargs):
        """Dynamically add/update elements to the buffer given as kwargs"""
        for (field, value) in kwargs.items():
            setattr(self, field, value)
